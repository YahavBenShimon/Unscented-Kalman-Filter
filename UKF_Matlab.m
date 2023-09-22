clear all
close all
clc

%% Initialize Parameters and Initial Conditions
x_0 = 100;%[m]
z_0 = 200;%[m]
v_x_0 = 2;%[m/s]
v_z_0 = 1;%[m/s]
theta_0 = 0.01;%[rad]
theta_dot_0 = 0.01;%[rad\s]

P = diag([10,15,2,3,0.08]).^2;
R = diag([8,4]).^2;
H = [1,0,0,0,0; 0,1,0,0,0];

dt = 0.01;
T = 40;
t_span = 0:dt:T+dt;

%% Solve Differential Equations
X_real = solve_differential_equations(x_0, v_x_0, z_0, v_z_0, theta_0, theta_dot_0, t_span);

%% Define GPS Vector
gps = define_gps_vector(X_real, R);

%% Perform Extended Kalman Filter
[S, X, P, Residual_m, Residual_k, std_k] = perform_unscented_kalman_filter...
    (dt,H,R,gps,t_span,P,X_real,x_0,v_x_0,z_0,v_z_0,theta_0,theta_dot_0);

%% Plot Results
plot_results(gps, X, X_real);


function X_real = solve_differential_equations...
    (x_0, v_x_0, z_0, v_z_0, theta_0, theta_dot_0, t_span)
    % Defining differential equations
    syms x(t) z(t) theta(t) Y
    Eq1 = ...
    diff(z,2) == 0.2*abs(cos(t))*cos(theta) - 0.4*abs(sin(t))*sin(theta);
    Eq2 = ...
    diff(x,2) == 0.4*abs(sin(t))*cos(theta) + 0.2*abs(cos(t))*sin(theta);
    Eq3 = ...
    diff(theta,1) == theta_dot_0; 

    % Turn equations into a vector form
    [Eqs_vector, ~] = odeToVectorField(Eq1, Eq2, Eq3);
    Eqs_Fcn = matlabFunction(Eqs_vector, 'vars', {'t', 'Y'});
    initCond = [x_0, v_x_0, z_0, v_z_0, theta_0];
    [T, Y] = ode45(Eqs_Fcn, t_span, initCond);

    X_real = zeros(5, length(T));
    X_real(1,:) = Y(:,1); % x real
    X_real(2,:) = Y(:,3); % z real
    X_real(3,:) = Y(:,2); % v_x real
    X_real(4,:) = Y(:,4); % v_z real
    X_real(5,:) = Y(:,5); % theta real
end

function gps = define_gps_vector(X_real, R)
    % Initalize gps vector
    gps = zeros(2, length(X_real(1,:)));
    
    % Define the gps matrix
    gps(1,:) = X_real(1,:) + sqrt(R(1,1)) .* randn(1, length(X_real(1,:)));
    gps(2,:) = X_real(2,:) + sqrt(R(2,2)) .* randn(1, length(X_real(2,:)));
end

function [S, X, P, Residual_m, Residual_k, std_k] = perform_unscented_kalman_filter...
    (dt,H,R,gps,t_span,P,X_real,x_0,v_x_0,z_0,v_z_0,theta_0,theta_dot_0)
    
    n = 5; % Number of States
    alpha = 1e-3; % Alpha parameter for sigma point generation
    beta = 2; % Beta parameter for sigma point generation
    kappa = 3 - n; % Kappa parameter for sigma point generation
    lambda = alpha^2 * (n + kappa) - n; % Scaling parameter
    
    X = zeros(n, length(t_span));
    X(:,1) = [x_0,z_0,v_x_0,v_z_0,theta_0]; % Initialize state
    std_k = zeros(n, length(t_span));
    std_k(:,1) = sqrt(diag(P));
    
    Wm = [lambda / (n + lambda), repmat(0.5 / (n + lambda), 1, 2 * n)]; % Weights for means
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta); % Weights for covariance
    
    for i = 1:length(t_span) - 1
        
        % Generate Sigma Points
        sqrtP = chol((n + lambda) * P)';
        sigma_pts = zeros(n, 2 * n + 1);
        sigma_pts(:, 1) = X(:, i);
        sigma_pts(:, 2:n+1) = bsxfun(@plus, X(:, i), sqrtP);
        sigma_pts(:, n+2:end) = bsxfun(@minus, X(:, i), sqrtP);
        
        % Time Update (Prediction)
        X_pred = zeros(n, 1);
        for j = 1:size(sigma_pts, 2)
            % Process model applied to each sigma point
            sp = sigma_pts(:, j);
            sp_transformed = sp + dt * [sp(3); sp(4); 0.4*abs(sin(t_span(i)))*cos(sp(5))+0.2*abs(cos(t_span(i)))*sin(sp(5)); ...
                                        0.2*abs(cos(t_span(i)))*cos(sp(5)) - 0.4*abs(sin(t_span(i)))*sin(sp(5)); theta_dot_0];
            X_pred = X_pred + Wm(j) * sp_transformed;
        end
        
        P_pred = zeros(n, n);
        for j = 1:size(sigma_pts, 2)
            % Calculate the Predicted Covariance
            sp = sigma_pts(:, j);
            sp_transformed = sp + dt * [sp(3); sp(4); 0.4*abs(sin(t_span(i)))*cos(sp(5))+0.2*abs(cos(t_span(i)))*sin(sp(5)); ...
                                        0.2*abs(cos(t_span(i)))*cos(sp(5)) - 0.4*abs(sin(t_span(i)))*sin(sp(5)); theta_dot_0];
            P_pred = P_pred + Wc(j) * (sp_transformed - X_pred) * (sp_transformed - X_pred)';
        end
        
        % Measurement Update
        z_pred = H * X_pred;
        Residual_m(:, i+1) = gps(:, i+1) - z_pred;
        
        S = H * P_pred * H' + R;
        K = P_pred * H' * inv(S);
        
        X(:, i+1) = X_pred + K * Residual_m(:, i+1);
        P = (eye(n) - K * H) * P_pred;
        
        Residual_k(:, i+1) = X_real(:, i+1) - X(:, i+1);
        std_k(:, i+1) = sqrt(diag(P));
    end
    
end
    
function plot_results(gps, X, X_real) 
        
    %X Vs Z
    figure('Name', 'Postion result');

    plot(gps(1,:),gps(2,:),'o',X(1,:),X(2,:),'*',X_real(1,:),X_real(2,:),'-k','LineWidth',0.8)
    xlabel('$X [m]$', 'Interpreter','latex')
    ylabel('$Z [m]$', 'Interpreter','latex')
    set(gca,'fontsize',16)
    box on
    legend('GPS Location','EKF Prediction','Trajectory')
    legend('Location','southeast')
    pbaspect([1 1 1])
end
