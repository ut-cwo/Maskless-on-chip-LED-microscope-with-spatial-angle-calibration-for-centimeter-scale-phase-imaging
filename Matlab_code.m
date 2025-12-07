clc
clear all; 
close all;

% Add paths to utility functions (forward model, TV prox, etc.)
addpath('.\Utility\unlocbox');
addpath('.\Utility\core_functions');

% Initialize the UNLocBoX (used for proximal operators like TV)
init_unlocbox();

% -------------------------------------------------------------------------
% Load measurement data
% -------------------------------------------------------------------------
load(".\FOV01\patch_12.mat")  % Choose the patch number to reconstruct

sliderDisplayImVC2(A_acqs, {'colormap gray','title(''Pixelated intensity acquisitions'')'});

% -------------------------------------------------------------------------
% Reconstruction / optimization hyperparameters
% -------------------------------------------------------------------------
usegpu    = false;       % Use GPU arrays if available
Amp_grad  = 1;          % Amplitude gradient scaling factor (used in gradient_update)
N_iter    = 100;        % Number of outer iterations (FISTA / gradient loop)
t_k       = 1;          % FISTA momentum variable
cost      = zeros(N_iter,1);  % To store cost per iteration
plot_range= [-0.1,0.1];       % Phase display range for MSBP_progview
step_size = 2;          % Gradient descent step size
regparam  = 1.5e-3;     % Regularization parameter for TV proximal operator

% -------------------------------------------------------------------------
% Illumination plane parameters
% (LED / plane-wave illumination model)
% -------------------------------------------------------------------------
lambda    = 632*10^-9;  % Wavelength [m]
di        = 0*10^-9;    % Propagation distance from illumination plane to object (here 0)

% -------------------------------------------------------------------------
% Image plane parameters
% (sensor side geometry)
% -------------------------------------------------------------------------
NA        = 1;                % Numerical aperture used for cropping in frequency domain
m         = size(A_acqs,1);   % Number of pixels in measured intensity (per side)

% -------------------------------------------------------------------------
% Object plane parameters
% -------------------------------------------------------------------------
n_imm     = 1;           % Immersion medium refractive index (not directly used here)
ps        = dpix/sig;    % Effective object-plane pixel size after upsampling
n         = m*sig;       % Object-plane pixel count per side

% -------------------------------------------------------------------------
% Illumination / frequency trajectory correction parameters
% -------------------------------------------------------------------------
N_illum         = size(A_acqs,3);    % Number of illumination angles / measurements

pos_shift_dir   = zeros(N_illum,2);  % (Unused here) for direction history of shifts
betaa           = 1*ones(N_illum,1); % Step scaling for position updates per angle
shift_max       = 0.2;               % Max allowable shift (pixels) in registration
iter_position   = 2;                 % Start updating positions after this iteration
iter_Stop       = 20;                % Stop updating positions after this iteration

% -------------------------------------------------------------------------
% Compute initial illumination spatial frequencies from Shift_x, Shift_y
% Shift_x, Shift_y are assumed to be in detector pixels
% fx_illum, fy_illum are spatial frequencies of plane-wave illumination
% -------------------------------------------------------------------------
fx_illum = (1/lambda) * (sin(atan(Shift_x*dpix/do)));  
fy_illum = (1/lambda) * (sin(atan(Shift_y*dpix/do))); 

% Plot initial estimated illumination frequency trajectory
figure
plot(fx_illum, fy_illum, 'r.'); 
axis equal; axis tight;
title('Estimated Illumination frequency trajectory');

% Fancy plot with both dots and connecting lines
figure
plot(fx_illum, fy_illum, 'r.', 'MarkerSize', 10); 
hold on;
plot(fx_illum, fy_illum, 'r-', 'LineWidth', 0.2); 

% Simple arrow annotation (in figure-normalized coordinates)
arrow_start = [min(fx_illum), max(fy_illum)];  % computed but not used below
arrow_end   = [max(fx_illum), min(fy_illum)];
% This arrow is placed in normalized figure coordinates [0,1]x[0,1]
annotation('arrow', [0, 0], [1, 1]);

% -------------------------------------------------------------------------
% Recompute some geometry variables (m, n) to ensure consistency
% -------------------------------------------------------------------------
m = size(A_acqs,1);
n = m*sig;

% -------------------------------------------------------------------------
% Padding parameters
% We compute how much to pad based on maximum illumination angles
% so that lateral shifts during propagation do not truncate the field.
% -------------------------------------------------------------------------
thetax_illum     = asin(-fx_illum*lambda);  % Illumination angles in x
thetay_illum     = asin(-fy_illum*lambda);  % Illumination angles in y

[thethax_max, ~] = max(thetax_illum);       % Max tilt in x
max_shift_x      = do * tan(thethax_max);   % Max lateral shift in x (meters)
max_shift_x_px   = ceil(max_shift_x / ps);  % in object-plane pixels

[thethay_max, ~] = max(thetay_illum);       % Max tilt in y
max_shift_y      = do * tan(thethay_max);   % Max lateral shift in y (meters)
max_shift_y_px   = ceil(max_shift_y / ps);  % in object-plane pixels

max_shift_px     = max(max_shift_y_px, max_shift_x_px);

% pdar1: padding needed for angular shift; pdar2: for Fresnel kernel support
pdar1      = max_shift_px + 10;    
kernelsize = do * lambda / (ps^2);                 % Effective Fresnel kernel size
pdar2      = ceil(0.5 * (kernelsize - n));         % Extra padding from kernel size

% Choose the worst-case padding
if pdar1 > pdar2
    pdar = pdar1;
    N    = n + 2*pdar;
end
if pdar1 < pdar2
    pdar = pdar2;
    N    = n + 2*pdar;
end

% -------------------------------------------------------------------------
% Define spatial (x,y) and frequency (Fx,Fy) grids for object plane
% -------------------------------------------------------------------------
x      = ps * (-N/2 : N/2-1);   % Object-plane coordinate grid
[X,Y]  = meshgrid(x, -x);       % Note: -x for Y to match coordinate convention

dfx    = 1/(N*ps);              % Frequency spacing
fx     = dfx * (-N/2 : N/2-1);  % Frequency grid
[Fx,Fy]= meshgrid(fx, -fx);     % Same sign swap for Fy

% Shift to put zero frequency in center
Fx     = ifftshift(Fx);
Fy     = ifftshift(Fy);

% Quantize illumination frequencies to the frequency grid
fx_illum = fix(fx_illum/dfx)*dfx;
fy_illum = fix(fy_illum/dfx)*dfx;

% -------------------------------------------------------------------------
% Propagation transfer functions
% Hi: propagation from illumination plane to object (distance di)
% Ho: propagation from object to sensor (distance do)
% Hoh: conjugate of Ho, used in gradient update
% NA_crop: logical mask for NA-limited frequencies
% -------------------------------------------------------------------------
prop_phs = 1i*2*pi*sqrt((1/lambda)^2 - (Fx.^2 + Fy.^2));  % Angular spectrum phase
NA_crop  = (Fx.^2 + Fy.^2) > (NA/lambda)^2;               % Outside NA region

Hi   = exp(prop_phs*di);
Ho   = exp(prop_phs*do);
Hoh  = conj(exp(prop_phs*do));

% -------------------------------------------------------------------------
% Initialize object estimate
% reconobj is the complex object field (amplitude & phase)
% reconobj_prox is the previous proximal point (for FISTA update)
% -------------------------------------------------------------------------
reconobj      = ones(N,N);       % Start with uniform amplitude=1, phase=0
reconobj_prox = reconobj;

% -------------------------------------------------------------------------
% Move data and operators to GPU, if requested
% -------------------------------------------------------------------------
if usegpu
    X             = gpuArray(X);
    Y             = gpuArray(Y);
    lambda        = gpuArray(lambda);
    di            = gpuArray(di);
    do            = gpuArray(do);
    fx_illum      = gpuArray(fx_illum);
    fy_illum      = gpuArray(fy_illum);
    Hi            = gpuArray(Hi);
    Ho            = gpuArray(Ho);
    Hoh           = gpuArray(Hoh);
    reconobj      = gpuArray(reconobj);
    reconobj_prox = gpuArray(reconobj_prox);
    A_acqs        = gpuArray(A_acqs);
end

% Count how many figures are open (for MSBP_progview indexing)
figHandles = findobj('Type', 'figure');
numFigures = numel(figHandles);

% Initial visualization of phase
figure
MSBP_progview(angle(reconobj), numFigures+1, plot_range, "Reconstruction")
caxis([-0.6 1.2])
pause(0.01);

% -------------------------------------------------------------------------
% Main iterative reconstruction loop (gradient descent + FISTA + TV prox)
% -------------------------------------------------------------------------
t_start = tic;
for iter = 1:N_iter

    % Initialize total gradient over all illuminations
    if usegpu
        grad_total = gpuArray(zeros(size(reconobj)));
    else
        grad_total = zeros(size(reconobj));
    end

    % Randomize order of illuminations each iteration
    for k = randperm(N_illum)
        
        % Optional: enforce unit amplitude (projection onto |obj|=1) in early iterations
        if iter < 45
            reconobj = exp(1j .* angle(reconobj));
        end

        % Gradient for this particular illumination
        if usegpu
        grad = gpuArray(zeros(size(reconobj)));
        else
            grad = (zeros(size(reconobj)));
        end

        % Plane wave illumination for k-th angle
        % Fk = exp(i 2π (fx_k X + fy_k Y))
        Fk = exp(1i * 2 * pi * (fx_illum(k) * X + fy_illum(k) * Y));

        % Forward model:
        %   Uk: complex field at sensor plane
        %   Qk: intermediate fields as needed by gradient_update
        [Uk, Qk] = Forward(reconobj, Fk, Hi, Ho, lambda, NA_crop);

        % Crop out padding and downsample to sensor resolution using S(.)
        Ik_forw = S(abs(Uk(pdar+1:end-pdar, pdar+1:end-pdar)).^2, sig, usegpu);
        Ak_forw = sqrt(Ik_forw);     % Forward amplitude

        % Measured amplitude for this illumination
        Ak_meas = A_acqs(:,:,k);

        % Compute gradient wrt object and cost for this illumination
        [grad, cost_k] = gradient_update(...
            Ak_meas, Ik_forw, Ak_forw, Hoh, pdar, sig, usegpu, ...
            Uk, Qk, Amp_grad, reconobj, lambda);

        % Accumulate gradient over all illuminations
        grad_total = grad_total + grad;

        % -----------------------------------------------------------------
        % Position / illumination frequency refinement via registration
        % -----------------------------------------------------------------
        if iter > iter_position && iter < iter_Stop
            % Register forward amplitude to measured amplitude using DFT-based
            % subpixel registration (dftregistration).
            [shift_diff, ~] = dftregistration(fft2(Ak_forw), fft2(Ak_meas), 1000);

            % shift_diff: [error, diffphase, row_shift, col_shift]
            shift_diff_x = shift_diff(4);   % shift along x
            shift_diff_y = -shift_diff(3);  % shift along y (sign difference)

            % Clamp shifts to max magnitude
            if abs(shift_diff_x) > abs(shift_max)
                shift_diff_x = sign(shift_diff_x) * shift_max;
            end
            if abs(shift_diff_y) > abs(shift_max)
                shift_diff_y = sign(shift_diff_y) * shift_max;
            end

            % Update estimated LED / illumination shifts
            Shift_x(k)  = Shift_x(k) - betaa(k)*shift_diff_x;
            Shift_y(k)  = Shift_y(k) - betaa(k)*shift_diff_y;

            % Convert updated shifts back to spatial frequencies
            fx_illum(k) = (1/lambda) * sin(atan(Shift_x(k)*dpix/do));  
            fy_illum(k) = (1/lambda) * sin(atan(Shift_y(k)*dpix/do)); 

            % Snap to available frequency grid
            fx_illum(k) = fix(fx_illum(k)/dfx)*dfx;
            fy_illum(k) = fix(fy_illum(k)/dfx)*dfx;

        end

        % Accumulate cost from this illumination
        cost(iter) = cost(iter) + cost_k;

        % Simple console print for monitoring
        fprintf('iteration: %d   error: %.3f   illum_angle: %d\n',iter, cost_k, k);
    end

    % ---------------------------------------------------------------------
    % Gradient descent update (average gradient over all illuminations)
    % ---------------------------------------------------------------------
    reconobj = reconobj - (step_size * (grad_total ./ N_illum));

    % TV proximal step (denoising / regularization on complex object)
    reconObj_prox1 = prox_tv(reconobj, regparam);

    % ---------------------------------------------------------------------
    % FISTA-style acceleration with adaptive restart
    % ---------------------------------------------------------------------
    if iter > 1
        % If current cost increased compared to previous iteration, restart
        if cost(iter) > cost(iter-1)
            t_k      = 1;
            reconobj = reconobj_prox;  % revert to previous proximal point
            continue;
        end
    end

    % FISTA momentum update
    t_k1  = 0.5 * (1 + sqrt(1 + 4 * t_k^2));
    beta  = (t_k - 1) / t_k1;

    % Accelerated update
    reconobj      = reconObj_prox1 + beta * (reconObj_prox1 - reconobj_prox);
    t_k           = t_k1;
    reconobj_prox = reconObj_prox1;

    % Optional projection: clip amplitude to <= 1 while keeping phase
    %reconobj = exp(1i * angle(reconobj)) .* min(abs(reconobj), 1);

    % ---------------------------------------------------------------------
    % Live visualization of phase during iterations
    % ---------------------------------------------------------------------
    MSBP_progview(angle(reconobj), numFigures+1, plot_range, "Reconstruction")
    caxis([-0.6 1.2])
    pause(0.01);

end

% -------------------------------------------------------------------------
% Timing and final reporting
% -------------------------------------------------------------------------
total_time = toc(t_start);  % total_time in seconds
fprintf('Total time for %d iterations: %.2f seconds (%.2f minutes)\n', ...
        N_iter, total_time, total_time/60);

% -------------------------------------------------------------------------
% Cost / loss plot
% -------------------------------------------------------------------------
figure
plot(1:N_iter, cost)
xlabel("Iteration")
ylabel("Cost")
title("Loss plot")

% -------------------------------------------------------------------------
% Display one raw measurement and final reconstruction (amplitude & phase)
% -------------------------------------------------------------------------
figure
imagesc(A_acqs(:,:,113))
colormap("gray")
title("Measured amplitude (illumination 113)")
colorbar 

figure
imagesc(abs(reconobj(pdar+1:end-pdar, pdar+1:end-pdar)))
colormap("gray")
title("Reconstructed amplitude")
colorbar 

figure
imagesc(angle(reconobj(pdar+1:end-pdar, pdar+1:end-pdar)))
colormap("gray")
title("Reconstructed phase")
colorbar 
caxis([-0.2 2])

% -------------------------------------------------------------------------
% Plot corrected illumination frequency trajectory (after registration)
% -------------------------------------------------------------------------
figure
plot(fx_illum, fy_illum, 'r.'); 
axis equal; axis tight;
title('Corrected  frequency trajectory');


function Inten = Readimagesfromfolder(folder_name,N1x,N2x,N1y,N2y)
if ~isdir(folder_name)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(folder_name, '*.tiff');
Files = dir(filePattern);
for k = 1:length(Files)
  baseFileName = Files(k).name;
  fullFileName = fullfile(folder_name, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  Im = double(imread(fullFileName));
  Inten(:,:,k)=Im(N1x:N2x,N1y:N2y);
end
end
function [Uk,Qk]= Forward(P,Fk,Hi,Ho,lambda,NA_crop)
    Fk          = Fk;
    Qk          = ifft2(fft2(Fk).* Hi);
    Sk          = Qk.* P;
    Uk          = ifft2(fft2(Sk).* Ho);
    Uk(NA_crop) = 0;
end
function [grad,cost] =gradient_update(Ak_meas,Ik_forw,Ak_forw, Hoh, pdar, sig, use_gpu, Uk,Qk,Amp_grad, P_est,lambda)

   Ek            = Ak_forw - Ak_meas;
   cost          = norm(Ek(:))^2;
   T1            = (1/2)*(Ek./Ak_forw);
   T1(isnan(T1)) = 0;
   T1(isinf(T1)) = 0;

   T2 =  ST(T1,sig);
   %T2            = imresize(T1, sig);
   T3            = T2.*Uk(pdar+1:end-pdar,pdar+1:end-pdar);
   T3            = padarray(T3,[pdar,pdar,0]); 
   T4            = ifft2(fft2(T3).*Hoh);
   T5            = (T4.*conj(Qk)) ;
   grad          = T5;
end
function u = S(x,sigma,use_gpu)
        u = zeros(size(x));
    if use_gpu
        u = gpuArray(u);
        x = gpuArray(x);
    else
        u = (u);
        x = (x);
    end

    for r = 0:sigma-1
        for c = 0:sigma-1
            u(1:sigma:end,1:sigma:end) = u(1:sigma:end,1:sigma:end) + x(1+r:sigma:end,1+c:sigma:end);
        end
    end
    u = u(1:sigma:end,1:sigma:end)./sigma^2;
end
function u = ST(x,sigma)
    u = zeros(size(x)*sigma);
    for r = 0:sigma-1
        for c = 0:sigma-1
            u(1+r:sigma:end,1+c:sigma:end) = x;
        end
    end
end
function sliderDisplayImVC2(data, figMod, circ)
% This function creates plots for data, a multidimensional array (3D or
% more).
% Sliders are created to navigate through data.
% 
% For example, let's assume data is 3D, containing images (in the first two 
% dimensions) for different values of defocus (third dimension). 
% sliderDisplay will display these images for a certain defocus value that
% can be changed with the slider.
% 
% For first time use, sliderDisplay() with no parameter will automatically
% generate data to plot.
% figMod is a string of figure commands (such as caxis, colorbar, etc.)
% 
% To adapt sliderDisplay to other situations, there are only two sections
% to modify: "Values to change" in the beginning and "Here is what you want
% to display" in the very end.

%% Data generation (only for testing)

if nargin == 0
    data = zeros(100,100,7,6);
    for i = 1:size(data,3)
        for j = 1:size(data,4)
            data(:,:,i,j) = rand(100);
        end
    end
    circ=zeros(size(data,3),6);
    figMod = '';
elseif nargin == 1
    circ=zeros(size(data,3),6);
    figMod = '';
elseif nargin == 2
    circ=zeros(size(data,3),6);
end

%% Values to change

if numel(size(circ))<3
    sliderNumber = size(size(circ),2) - 1;
else
    sliderNumber = size(size(circ),2) - 2;
end
sliderDimension1 = 1;  % Dimension corresponding to Slider 1 (3 for the example above)
sliderDimension2 = 4;  % Same for Slider 2, won't be used if nSlider < 2
sliderDimension3 = 5;  % Same for Slider 3, won't be used if nSlider < 3

screenWidth = 1920;
sliderLeftPosition = 0;  % Change to round(5*screenWidth/12) if it covers the image
sliderLength = round(screenWidth/6);

% For display
sliderName1 = 'Time';  % Description of sliderDimension1 (defocus for the example above)
sliderName2 = '';
sliderName3 = '';
sliderStep1 = 1;  % Step for each dimension (defocus distance between two images for the example above)
sliderStep2 = 1;
sliderStep3 = 1;

%% Other initializations

sliderTotal1 = size(circ,sliderDimension1);
sliderTotal2 = size(circ,sliderDimension2);
sliderTotal3 = size(circ,sliderDimension3);
if strcmp(sliderName1,'')
    sliderName1 = 'Slider 1 position';
end
if strcmp(sliderName2,'')
    sliderName2 = 'Slider 2 position';
end
if strcmp(sliderName3,'')
    sliderName3 = 'Slider 3 position';
end

%% Display initialization

currentPosition1 = 1;
currentPosition2 = 1;
currentPosition3 = 1;
lastPosition1 = currentPosition1;
lastPosition2 = currentPosition2;
lastPosition3 = currentPosition3;

figure();

%% Slider and text creation

if sliderNumber >= 1
    handleSlider1 = uicontrol('Style', 'slider',...
        'Min',1,'Max',sliderTotal1,'Value',currentPosition1,...
        'sliderStep', [min(1/(sliderTotal1-1),0.1) 0.2], ...
        'Position', [sliderLeftPosition 25 sliderLength 25]);
    addlistener(handleSlider1,'ContinuousValueChange',@callback1);
    
    % Text
    handleText1 = uicontrol('Style','text',...
        'Position',[sliderLeftPosition 0 sliderLength 25],...
        'String',[sliderName1 ': ' num2str(currentPosition1*sliderStep1)]);
end
if sliderNumber >= 2
    handleSlider2 = uicontrol('Style', 'slider',...
        'Min',1,'Max',sliderTotal2,'Value',currentPosition2,...
        'sliderStep', [min(1/(sliderTotal2-1),0.1) 0.2], ...
        'Position', [sliderLeftPosition 75 sliderLength 25]);
    addlistener(handleSlider2,'ContinuousValueChange',@callback2);
    
    % Text
    handleText2 = uicontrol('Style','text',...
        'Position',[sliderLeftPosition 50 sliderLength 25],...
        'String',[sliderName2 ': ' num2str(currentPosition2*sliderStep2)]);
end
if sliderNumber >= 3
    handleSlider3 = uicontrol('Style', 'slider',...
        'Min',1,'Max',sliderTotal3,'Value',currentPosition3,...
        'sliderStep', [min(1/(sliderTotal3-1),0.1) 0.2], ...
        'Position', [sliderLeftPosition 125 sliderLength 25]);
    addlistener(handleSlider3,'ContinuousValueChange',@callback3);
    
    % Text
    handleText3 = uicontrol('Style','text',...
        'Position',[sliderLeftPosition 100 sliderLength 25],...
        'String',[sliderName3 ': ' num2str(currentPosition3*sliderStep3)]);
end

%% Display

display();

%% Functions called when slider is moved

    function callback1(varargin)
        currentPosition1 = round(get(handleSlider1,'Value'));
        set(handleSlider1,'Value',currentPosition1);
        
        if currentPosition1 ~= lastPosition1
            currentPosition2 = lastPosition2;
            currentPosition3 = lastPosition3;
            
            display();
            
            lastPosition1 = currentPosition1;
        end
    end
    function callback2(varargin)
        currentPosition2 = round(get(handleSlider2,'Value'));
        set(handleSlider2,'Value',currentPosition2);
        
        if currentPosition2 ~= lastPosition2
            currentPosition1 = lastPosition1;
            currentPosition3 = lastPosition3;
            
            display();
            
            lastPosition2 = currentPosition2;
        end
    end
    function callback3(varargin)
        currentPosition3 = round(get(handleSlider3,'Value'));
        set(handleSlider3,'Value',currentPosition3);
        
        if currentPosition3 ~= lastPosition3
            currentPosition1 = lastPosition1;
            currentPosition2 = lastPosition2;
            
            display();
            
            lastPosition3 = currentPosition3;
        end
    end

%% Display

    function display()
        % Change text
        if sliderNumber >= 1
            delete(handleText1);
            handleText1 = uicontrol('Style','text',...
                'Position',[sliderLeftPosition 0 sliderLength 25],...
                'String',[sliderName1 ': ' num2str(currentPosition1*sliderStep1)]);
        end
        if sliderNumber >= 2
            delete(handleText2);
            handleText2 = uicontrol('Style','text',...
                'Position',[sliderLeftPosition 50 sliderLength 25],...
                'String',[sliderName2 ': ' num2str(currentPosition2*sliderStep2)]);
        end
        if sliderNumber >= 3
            delete(handleText3);
            handleText3 = uicontrol('Style','text',...
                'Position',[sliderLeftPosition 100 sliderLength 25],...
                'String',[sliderName3 ': ' num2str(currentPosition3*sliderStep3)]);
        end
        
        cla();  % Clear axes
        
        %% Here is what you want to display
        % currentPosition1 represents the current value for
        % sliderDimension1. For the example above, to display images at
        % a given defocus distance, use:
        %     imshow(data(:,:,currentPosition1));
        
        imagesc(squeeze(data(:,:,currentPosition1)));
        %imagesc(squeeze(data(:,:,currentPosition1,currentPosition2,currentPosition3)));
        col={'r','g','m','b','c','w','y','k','r','g','m','b','c'};
%         axis(451+100*[-1 1 -1 1]); 
        axis image; 
        colorbar
        hold on;
        for kk=1:size(circ,3)
            viscircles(circ(currentPosition1,1:2,kk,currentPosition2,currentPosition3),circ(currentPosition1,3,kk,currentPosition2,currentPosition3),'EdgeColor',col{kk},'LineWidth',0.25);
        end
        
        for ii=1:length(figMod)
            eval(figMod{ii});
        end
        
        
    end
end


function MSBP_progview(obj,fignum,plot_range,titletext)

figure(fignum)
imagesc(obj); 
colormap gray; 
title(titletext);
%caxis(plot_range);
colorbar
end