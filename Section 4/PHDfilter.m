classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            dummy_states = struct('x', obj.paras.states(1).x, 'P', obj.paras.states(1).P);

            % Predicting birth targets
            for i=1:length(birthmodel)
                obj.paras.w = cat(1, obj.paras.w, birthmodel(i).w);
                dummy_states.x = motionmodel.f(birthmodel(i).x);
                dummy_states.P = motionmodel.F(0) * birthmodel(i).P * motionmodel.F(0)' + motionmodel.Q;
                obj.paras.states = cat(1, obj.paras.states, dummy_states);
                % obj.paras.states = cat(1, obj.paras.states, struct('x', birthmodel(i).x, 'P', birthmodel(i).P));
            end

            % Predicting exist targets
            for i=1:length(obj.paras.w)
                obj.paras.w = cat(1, obj.paras.w, obj.paras.w(i) + log(P_S));
                dummy_states.x = motionmodel.f(obj.paras.states(i).x);
                dummy_states.P = motionmodel.F(0) * obj.paras.states(i).P * motionmodel.F(0)' + motionmodel.Q;
                obj.paras.states = cat(1, obj.paras.states, dummy_states);
            end
            % for i=1:length(birthmodel)
            %     obj.paras.w = cat(1, obj.paras.w, birthmodel(i).w);
            %     obj.paras.states = cat(1, obj.paras.states, struct('x', birthmodel(i).x, 'P', birthmodel(i).P));
            % end
           
        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            temp_obj_w = [];
            temp_obj_states = [];
            dummy_states = struct('x', obj.paras.states(1).x, 'P', obj.paras.states(1).P);
            hypotheses_num = length(obj.paras.w);
            state_dim = size(obj.paras.states(1).x, 1);
            meas_dim = size(z,1);
            
            % Get weights and paras for miss detection
            for i=1:hypotheses_num
                temp_obj_w = [temp_obj_w, log(1 - P_D) + obj.paras.w(i)];
                dummy_states.x = obj.paras.states(i).x;
                dummy_states.P = obj.paras.states(i).P;
                temp_obj_states = [temp_obj_states, dummy_states];
            end

            % Get weights and paras for detection
            % Perform ellipsoidal gating
            z_idx_ingates = false(length(z),1);
            for i=1:hypotheses_num
                dummy_states.x = obj.paras.states(i).x;
                dummy_states.P = obj.paras.states(i).P;
                [~, z_idx_ingates_curr] = GaussianDensity.ellipsoidalGating(dummy_states, z, measmodel, gating.size);
                z_idx_ingates = z_idx_ingates | z_idx_ingates_curr;
            end
            
            z_ingates = z(:, z_idx_ingates');
            
            % perform update 
            z_predicted = [];
            S = [];
            K = [];
            P = [];

            % Kalman update
            for i=1:hypotheses_num
                dummy_states.x = obj.paras.states(i).x;
                dummy_states.P = obj.paras.states(i).P;
                z_predicted = cat(2, z_predicted, measmodel.h(dummy_states.x));
                S = cat(3, S, measmodel.R + measmodel.H(0) * dummy_states.P * measmodel.H(0)');
                K = cat(3, K, dummy_states.P * measmodel.H(0)' / S(:, :, i));
                P = cat(3, P, (eye(state_dim) - K(:, :, i) * measmodel.H(0)) * dummy_states.P);
                % P = cat(3, P, dummy_states.P - K(:, :, i) * S(:, :, i) * K(:, :, i)');
                % P = cat(3, P, dummy_states.P - K(:, :, i) * measmodel.H(0) * dummy_states.P');
            end

            % temp_obj_w_unnormal = [];

            for i=1:size(z_ingates, 2)

                temp_obj_w_unnormal = [];

                for h=1:hypotheses_num
                    dummy_states.x = obj.paras.states(h).x + K(:, :, h) * (z_ingates(:, i) - z_predicted(:, h));
                    dummy_states.P = P(:, :, h);
                    temp_obj_states = [temp_obj_states, dummy_states];
                    temp_obj_w_unnormal = cat(2, temp_obj_w_unnormal, ...
                        log(P_D) + obj.paras.w(h) + log_mvnpdf(z_ingates(:, i), z_predicted(:, h), S(:, :, h)));
                end
                
                % Find the maximum value in the vector x
                max_w = max(temp_obj_w_unnormal);
                
                % Compute the log-sum-exp
                log_sum_w = max_w + log(sum(exp(temp_obj_w_unnormal - max_w)));

                for h=1:hypotheses_num
                    temp_obj_w = cat(2, temp_obj_w, temp_obj_w_unnormal(:, h) - log(intensity_c + exp(log_sum_w)));
                end
            end

            obj.paras.states = reshape(temp_obj_states, length(temp_obj_w), 1);
            obj.paras.w = reshape(temp_obj_w, length(temp_obj_w), 1);
        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            estimates = [];

            n = round(sum(exp(obj.paras.w)));
            [~, idx] = sort(obj.paras.w, 'descend');
            for best_idx=1:min(n, length(idx))
                estimates = [estimates, obj.paras.states(idx(best_idx)).x];
            end

        end
        
    end
    
end