classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
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
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
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
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1   

        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: array of size (total tracking time, object 
            %       state dimension)

            % In this implementation, everything will put into log scale
            % Initialization. At first step, weight is initialized in log scale,
            T = length(Z);
            weights = [0];
            hypotheses = [struct('x', state.x, 'P', state.P)];
            estimates = [];
            for t=1:T
                % Initialize weights and hypothesis
                propose_hypotheses = [];
                propose_weights = [];
                % Current observation
                z = cell2mat(Z(t));
                % Update step and get hypotheses with corresponding weight
                
                for past_idx=1:length(hypotheses)
                    % When measurements have object detection
                    % Perform ellipsoidal gating and only create object detection hypotheses for detections inside the gate;
                    [z_ingate, ~] = GaussianDensity.ellipsoidalGating(hypotheses(past_idx), z, measmodel, obj.gating.size);

                    % Probability also return in log scale
                    z_prob = GaussianDensity.predictedLikelihood(hypotheses(past_idx), z_ingate, measmodel);

                    for obs_idx=1:size(z_ingate, 2)
                        % Weights is also log scale
                        propose_weights = [propose_weights; 
                            weights(past_idx) + log(sensormodel.P_D) + z_prob(obs_idx) - log(sensormodel.lambda_c)];
                        dummy= GaussianDensity.update(hypotheses(past_idx), z_ingate(obs_idx), measmodel);
                        propose_hypotheses = [propose_hypotheses; dummy];
                    end

                    % When every measurements are cluster. Weights is also log scale
                    propose_weights = [propose_weights; weights(past_idx) + log(1 - sensormodel.P_D)];
                    dummy = hypotheses(past_idx);
                    propose_hypotheses = [propose_hypotheses; dummy];
                end

                % normalise hypothsis weights in log scale;
                propose_weights = renormalizeLogWeights(propose_weights);

                % prune hypotheses with small weights, and then re-normalise the weights;
                [propose_weights, propose_hypotheses] = hypothesisReduction.prune(propose_weights, propose_hypotheses, obj.reduction.w_min);
                propose_weights = renormalizeLogWeights(propose_weights);

                % hypothesis merging (to achieve this, you just need to directly call function hypothesisReduction.merge.);
                [propose_weights, propose_hypotheses] = hypothesisReduction.merge(propose_weights, propose_hypotheses, obj.reduction.merging_threshold, GaussianDensity);
                
                % cap the number of the hypotheses, and then re-normalise the weights;
                [propose_weights, propose_hypotheses] = hypothesisReduction.cap(propose_weights, propose_hypotheses, obj.reduction.M);
                propose_weights = renormalizeLogWeights(propose_weights);

                % extract object state estimate using the most probably hypothesis estimation;

                [~, best_state_idx] = max(propose_weights);
   
                estimates = [estimates, propose_hypotheses(best_state_idx).x];
                % for each hypothesis, perform prediction.
                weights = propose_weights;
                hypotheses = [];
                for curr_idx=1:length(propose_hypotheses)
                    hypotheses = [hypotheses; GaussianDensity.predict(propose_hypotheses(curr_idx), motionmodel)];
                end
            end
        end
        
    end
end

function normalized_log_weights = renormalizeLogWeights(log_weights)
    % Calculate the maximum log weight for numerical stability
    max_log_weight = max(log_weights);
    
    % Calculate the LogSumExp (LSE)
    LSE = max_log_weight + log(sum(exp(log_weights - max_log_weight)));
    
    % Renormalize the log weights by subtracting the LSE
    normalized_log_weights = log_weights - LSE;
end

function cellArray = convertToCell(GaussianSumEstimates)
    % Get the number of columns (T) from the input array
    [numRows, numCols] = size(GaussianSumEstimates);
    
    % Initialize the cell array
    cellArray = cell(numCols, 1);
    
    % Loop through each column and assign it to a cell
    for t = 1:numCols
        cellArray{t} = GaussianSumEstimates(:, t);
    end
end
