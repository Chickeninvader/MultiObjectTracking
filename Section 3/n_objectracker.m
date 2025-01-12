classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)

            % Each object contains hypotheses
            % lookup_table is an array contain idx of the local hypothesis
            % associate to object i, size (num of hypothesis, number of object)
            % local_hypotheses is a cell array contains object.
            % Each object is a cell array of struct contain x and P of
            % each local hypothesis

            % Initialization
            T = length(Z);  % Total tracking time
            obj_num = length(states);  % Number of objects
            estimates = cell(T, 1);  % Initialize estimates
            log_ws = ones(1, 1);  % Initialize log weights
            local_hypotheses = cell(obj_num, 1);  % Initialize local hypotheses
            lookup_table = ones(1, obj_num);  % Initialize lookup table
            M = 10;  % Hypothesis cap limit

            % Initialize local hypotheses for each object
            for i=1:obj_num
                local_hypotheses{i} = [states(i)];
            end

            for t=1:T
                z = Z{t};  % Current observation
                m_k = size(z, 2);  % Number of measurements
        
                % Local update: update hypotheses and weights for each object
                temp_local_hypotheses = cell(obj_num, 1);
                temp_local_log_ws = cell(obj_num, 1);
                for i = 1:obj_num
                    [temp_local_log_ws{i}, temp_local_hypotheses{i}] = obj.local_update(sensormodel.P_D, local_hypotheses{i}, z, measmodel, sensormodel);
                end

                % Global update
                [log_ws, lookup_table] = obj.lookup_table_update(log_ws, lookup_table, m_k, obj_num, M, temp_local_log_ws);

                % Reduction
                % Prune and cap global hypotheses
                [log_ws, lookup_table] = hypothesisReduction.prune(log_ws, lookup_table, obj.reduction.w_min);
                [log_ws, lookup_table] = hypothesisReduction.cap(log_ws, lookup_table, obj.reduction.M);

                % Prune local hypothesis that is not in global
                % hypotheses and reindex lookup table
                for i=1:obj_num
                    idx_in_global_hypothesis = unique(lookup_table(:, i));
                    local_hypotheses{i} = temp_local_hypotheses{i}(idx_in_global_hypothesis);
                    dummy_map = zeros(m_k+1, 1);
                    for idx=1:length(idx_in_global_hypothesis)
                        dummy_map(idx_in_global_hypothesis(idx)) = idx;
                    end
                    lookup_table(:, i) = dummy_map(lookup_table(:, i));
                end

                % extract object state estimates from the global hypothesis with the highest weight;
                [~, best_global_idx] = max(log_ws);
                estimates{t} = [];
                for i=1:obj_num
                    estimates{t} = [estimates{t}, local_hypotheses{i}(lookup_table(best_global_idx, i)).x];
                end
                
                % Prediction: curr_hypotheses now still contains obj_num
                % hypothesis
                local_hypotheses = obj.prediction(local_hypotheses, obj_num, motionmodel);
            end
            
        end

        function curr_hypotheses = prediction(obj, hypotheses, obj_num, motionmodel)
            % PREDICTION Predicts the next state of each hypothesis for all objects
            % INPUT:
            %   obj: an instance of the class (not used in this function, but typically part of method signature)
            %   hypotheses: cell array of size (number of objects, 1)
            %       Each cell contains an array of struct with fields:
            %           x: state mean vector
            %           P: state covariance matrix
            %   obj_num: number of objects
            %   motionmodel: struct with motion model functions
            %       f: function handle for state transition
            %       F: function handle for state transition Jacobian
            %       Q: process noise covariance matrix
            % OUTPUT:
            %   curr_hypotheses: cell array of size (number of objects, 1)
            %       Each cell contains an array of struct with predicted state mean and covariance
        
            % Initialize the cell array for current hypotheses
            curr_hypotheses = cell(obj_num, 1);
        
            % Iterate over each object
            for i = 1:obj_num
                % Get the hypotheses for the current object
                temp_hypothesis = hypotheses{i}; 
                new_hypothesis = []; % Pre-allocate array for new hypotheses
        
                % Iterate over each hypothesis for the current object
                for h = 1:length(temp_hypothesis)
                    % Predict the next state mean using the motion model function
                    dummy_states.x = motionmodel.f(temp_hypothesis(h).x);
        
                    % Predict the next state covariance using the motion model Jacobian and process noise
                    dummy_states.P = motionmodel.F(temp_hypothesis(h).x) * temp_hypothesis(h).P * motionmodel.F(temp_hypothesis(h).x)' + motionmodel.Q;
        
                    % Append the new hypothesis to the list
                    new_hypothesis = [new_hypothesis; dummy_states];
                end
        
                % Assign the new hypotheses to the current object
                curr_hypotheses{i} = new_hypothesis;
            end
        end

        function [log_w, curr_hypothesis] = local_update(obj, P_D, hypothesis, z, measmodel, sensormodel)
            % LOCAL_UPDATE performs the local update step for each local object hypothesis given measurements
            % INPUT:
            %   obj: an instance of the class (contains gating size and density update methods)
            %   P_D: probability of detection
            %   hypothesis: array of structs, each containing:
            %       x: state mean vector
            %       P: state covariance matrix
            %   z: matrix of measurements, size (measurement dimension, number of measurements)
            %   measmodel: struct with measurement model functions:
            %       h: function handle for measurement prediction
            %       H: function handle for measurement Jacobian
            %       R: measurement noise covariance matrix
            %   sensormodel: struct with sensor model parameters:
            %       lambda_c: clutter rate
            % OUTPUT:
            %   log_w: array of log weights for each updated hypothesis, size (hypothesis_num * (m_k + 1), 1)
            %   curr_hypothesis: array of updated hypotheses, size (hypothesis_num * (m_k + 1), 1)
            
            hypothesis_num = length(hypothesis);  % Number of hypotheses
            m_k = size(z, 2);  % Number of measurements
            dummy_states = struct('x', [], 'P', []);  % Initialize a dummy state struct
            curr_hypothesis = repmat(dummy_states, hypothesis_num * (m_k + 1), 1);  % Preallocate the current hypothesis array
            log_w = zeros(hypothesis_num * (m_k + 1), 1);  % Initialize log weights
        
            % Iterate over each hypothesis
            for h = 1:hypothesis_num
                % Perform ellipsoidal gating to determine which measurements are in the gate
                [~, z_idx_ingates] = GaussianDensity.ellipsoidalGating(hypothesis(h), z, measmodel, obj.gating.size);
        
                % Iterate over each possible measurement and the miss detection hypothesis
                for j = 1:m_k + 1
                    % If j is not the miss detection index and the measurement is not in the gate,
                    % set log_w to be inf and retain the old state
                    if j ~= m_k + 1 && not(z_idx_ingates(j))
                        dummy_states = hypothesis(h);
                        dummy_log_w = inf;
                    % If j corresponds to a detection at measurement j
                    elseif j ~= m_k + 1
                        dummy_states = obj.density.update(hypothesis(h), z(:, j), measmodel);
                        z_predicted = measmodel.h(hypothesis(h).x);
                        innovation_cov = measmodel.H(hypothesis(h).x) * hypothesis(h).P * measmodel.H(hypothesis(h).x)' + measmodel.R;
                        
                        % Ensure innovation_cov is symmetric
                        innovation_cov = (innovation_cov + innovation_cov') / 2;
                
                        % Ensure innovation_cov is positive definite
                        epsilon = 1e-6;
                        innovation_cov = innovation_cov + epsilon * eye(size(innovation_cov));
                        
                        % Calculate the log weight for the detection hypothesis
                        dummy_log_w = log(P_D) + log_mvnpdf(z(:, j), z_predicted, innovation_cov) - log(sensormodel.intensity_c);
                    % If j corresponds to a miss detection
                    else
                        dummy_states = hypothesis(h);
                        dummy_log_w = log(1 - P_D);
                    end
        
                    % Update hypothesis and weight
                    curr_hypothesis((h - 1) * (m_k + 1) + j) = dummy_states;
                    log_w((h - 1) * (m_k + 1) + j) = dummy_log_w;
                end
            end
        end

        function [update_log_ws, update_lookup_table] = lookup_table_update(obj, log_ws, lookup_table, m_k, obj_num, M, local_log_ws)
            hypotheses_num = length(log_ws);
            update_log_ws = [];
            update_lookup_table = [];

            for h=1:hypotheses_num
                L = inf(obj_num, m_k+obj_num);
                for i=1:obj_num
                    local_hypothesis_idx = lookup_table(h, i);
                    L(i, 1:m_k) = local_log_ws{i}((local_hypothesis_idx - 1) * (m_k + 1) + 1: (local_hypothesis_idx - 1) * (m_k + 1) + m_k);
                    L(i, m_k + i) = local_log_ws{i}(local_hypothesis_idx * (m_k + 1));
                end
                [col4rowBest,row4colBest,gainBest] = kBest2DAssign(L,M);
                for m=1:min(M, size(col4rowBest, 2))
                    update_log_w = log_ws(h) + gainBest(m);
                    update_lookup_table_row = col4rowBest(:,m)';
                    miss_detection_idx = update_lookup_table_row > m_k;
                    update_lookup_table_row(miss_detection_idx) = m_k+1;
                    
                    update_log_ws = [update_log_ws; update_log_w];
                    update_lookup_table = [update_lookup_table; update_lookup_table_row];
                end
            end
            [update_log_ws, ~] = normalizeLogWeights(update_log_ws);
        end
    end
end
