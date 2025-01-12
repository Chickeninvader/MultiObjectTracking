classdef PMBMfilter
    %PMBMFILTER is a class containing necessary functions to implement the
    %PMBM filter
    %Model structures need to be called:
    %    sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    properties
        density %density class handle
        paras   %%parameters specify a PMBM
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PMBMfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture)
            %       of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.PPP.w: weights of mixture components in PPP
            %       intensity --- vector of size (number of mixture
            %       components x 1) in logarithmic scale
            %       obj.paras.PPP.states: parameters of mixture components
            %       in PPP intensity struct array of size (number of
            %       mixture components x 1)
            %       obj.paras.MBM.w: weights of MBs --- vector of size
            %       (number of MBs (global hypotheses) x 1) in logarithmic 
            %       scale
            %       obj.paras.MBM.ht: hypothesis table --- matrix of size
            %       (number of global hypotheses x number of hypothesis
            %       trees). Entry (h,i) indicates that the (h,i)th local
            %       hypothesis in the ith hypothesis tree is included in
            %       the hth global hypothesis. If entry (h,i) is zero, then
            %       no local hypothesis from the ith hypothesis tree is
            %       included in the hth global hypothesis.
            %       obj.paras.MBM.tt: local hypotheses --- cell of size
            %       (number of hypothesis trees x 1). The ith cell contains
            %       local hypotheses in struct form of size (number of
            %       local hypotheses in the ith hypothesis tree x 1). Each
            %       struct has two fields: r: probability of existence;
            %       state: parameters specifying the object density
            
            obj.density = density_class_handle;
            obj.paras.PPP.w = [birthmodel.w]';
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = {};
        end
        
        function Bern = Bern_predict(obj,Bern,motionmodel,P_S)
            %BERN_PREDICT performs prediction step for a Bernoulli component
            %INPUT: Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar;
            %                          state: a struct contains parameters
            %                          describing the object pdf
            %       P_S: object survival probability

            % h_k = length(obj.paras.MBM.tt);
            % update_tt = {};
            % for h=1:h_k
            %     n_k = length(obj.paras.MBM.tt(h));
            %     past_local_hypothesis = obj.paras.MBM.tt(h);
            %     update_tt(h) = [];
            %     for i=1:n_k
            %         dummy_r = past_local_hypothesis(i).r * P_S;
            %         dummy_states = obj.density.predict(past_local_hypothesis(i), motionmodel);
            %         update_tt(h) = [update_tt(h); struct('r', dummy_r, 'state', dummy_states)];
            %     end
            % end
            % obj.paras.MBM.tt = update_tt;

            Bern.r = Bern.r * P_S;
            Bern.state = obj.density.predict(Bern.state, motionmodel);
            
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            %PPP_PREDICT performs predicion step for PPP components
            %hypothesising undetected objects.
            %INPUT: P_S: object survival probability --- scalar  
            % Predicting exist targets
            update_PPP.w = [];
            update_PPP.states = [];
            for i=1:length(obj.paras.PPP.w)
                update_PPP.w = [update_PPP.w; obj.paras.PPP.w(i) + log(P_S)];
                update_PPP.states = [update_PPP.states; obj.density.predict(obj.paras.PPP.states(i), motionmodel)];
            end
            % Predicting birth targets
            for i=1:length(birthmodel)
                update_PPP.w = [update_PPP.w; birthmodel(i).w];
                dummy_states = rmfield(birthmodel,'w')';
                update_PPP.states = [update_PPP.states; dummy_states(i)];
            end

            obj.paras.PPP = update_PPP;
        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D)
            %BERN_UNDETECTED_UPDATE calculates the likelihood of missed
            %detection, and creates new local hypotheses due to missed
            %detection.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth local
            %       hypothesis in the ith hypothesis tree. 
            %       P_D: object detection probability --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %       with fields: r: probability of existence --- scalar;
            %                    state: a struct contains parameters
            %                    describing the object pdf
            %       lik_undetected: missed detection likelihood --- scalar
            %       in logorithmic scale
            dummy_state_w = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            Bern.r = dummy_state_w.r * (1 - P_D) / (1 - dummy_state_w.r + dummy_state_w.r * (1-P_D));
            Bern.state = dummy_state_w.state;
            lik_undetected = log(1 - dummy_state_w.r + dummy_state_w.r * (1-P_D));
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            %BERN_DETECTED_UPDATE_LIK calculates the predicted likelihood
            %for a given local hypothesis. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth
            %       local hypothesis in the ith hypothesis tree.
            %       z: measurement array --- (measurement dimension x
            %       number of measurements)
            %       P_D: object detection probability --- scalar
            %OUTPUT:lik_detected: predicted likelihood --- (number of
            %measurements x 1) array in logarithmic scale 
            dummy_state_w = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            predict_x = measmodel.h(dummy_state_w.state.x);
            predict_P = measmodel.H(dummy_state_w.state.x) * dummy_state_w.state.P * measmodel.H(dummy_state_w.state.x)' + measmodel.R;
            predict_P = (predict_P + predict_P') / 2;
            lik_detected = [];
            for z_idx=1:size(z, 2)
                lik_detected = [lik_detected; log(dummy_state_w.r) + log_mvnpdf(z(:, z_idx), predict_x, predict_P) + log(P_D)];
            end
        end
        
        function Bern = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            %BERN_DETECTED_UPDATE_STATE creates the new local hypothesis
            %due to measurement update. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %                 local hypotheses. (i,j) indicates the jth
            %                 local hypothesis in the ith hypothesis tree.
            %       z: measurement vector --- (measurement dimension x 1)
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar; 
            %                          state: a struct contains parameters
            %                          describing the object pdf 
            dummy_state_w = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            Bern.r = 1;
            Bern.state = obj.density.update(dummy_state_w.state, z, measmodel);
        end
 
        function [Bern, lik_new] = PPP_detected_update(obj,indices,z,measmodel,P_D,clutter_intensity)
            %PPP_DETECTED_UPDATE creates a new local hypothesis by
            %updating the PPP with measurement and calculates the
            %corresponding likelihood.
            %INPUT: z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %       clutter_intensity: Poisson clutter intensity --- scalar
            %       indices: boolean vector, if measurement z is inside the
            %       gate of mixture component i, then indices(i) = true
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %             scalar;
            %             state: a struct contains parameters describing
            %             the object pdf
            %       lik_new: predicted likelihood of PPP --- scalar in
            %       logarithmic scale 
            lik_pred = 0;
            dummy_states = [];
            dummy_ws = [];
            for i=1:length(indices)
                if indices(i)
                    dummy_states = [dummy_states; obj.density.update(obj.paras.PPP.states(i), z, measmodel)];
                    obs_mean = measmodel.h(obj.paras.PPP.states(i).x);
                    obs_cov = measmodel.H(obj.paras.PPP.states(i).x) * obj.paras.PPP.states(i).P * measmodel.H(obj.paras.PPP.states(i).x)' + measmodel.R;
                    obs_cov = (obs_cov + obs_cov') / 2;
                    lik_pred = lik_pred + P_D * exp(obj.paras.PPP.w(i)) * exp(log_mvnpdf(z, obs_mean, obs_cov));
                    dummy_ws = [dummy_ws; obj.paras.PPP.w(i) + log(P_D) + log_mvnpdf(z, obs_mean, obs_cov)];
                end
            end
            lik_new = log(clutter_intensity + lik_pred);
            Bern.r = lik_pred / (clutter_intensity + lik_pred);
            Bern.state = obj.density.momentMatching(normalizeLogWeights(dummy_ws), dummy_states);
        end
        
        function obj = PPP_undetected_update(obj,P_D)
            %PPP_UNDETECTED_UPDATE performs PPP update for missed detection.
            %INPUT: P_D: object detection probability --- scalar
            for i=1:length(obj.paras.PPP.w)
                obj.paras.PPP.w(i) = obj.paras.PPP.w(i) + log(1-P_D);
            end
        end
        
        function obj = PPP_reduction(obj,prune_threshold,merging_threshold)
            %PPP_REDUCTION truncates mixture components in the PPP
            %intensity by pruning and merging
            %INPUT: prune_threshold: pruning threshold --- scalar in
            %       logarithmic scale
            %       merging_threshold: merging threshold --- scalar
            [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.prune(obj.paras.PPP.w, obj.paras.PPP.states, prune_threshold);
            if ~isempty(obj.paras.PPP.w)
                [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.merge(obj.paras.PPP.w, obj.paras.PPP.states, merging_threshold, obj.density);
            end
        end
        
        function obj = Bern_recycle(obj,prune_threshold,recycle_threshold)
            %BERN_RECYCLE recycles Bernoulli components with small
            %probability of existence, adds them to the PPP component, and
            %re-index the hypothesis table. If a hypothesis tree contains no
            %local hypothesis after pruning, this tree is removed. After
            %recycling, merge similar Gaussian components in the PPP
            %intensity
            %INPUT: prune_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold are pruned ---
            %       scalar
            %       recycle_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold needed to be
            %       recycled --- scalar
            
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold & x.r>=prune_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Here, we should also consider the weights of different MBs
                    idx_t = find(idx);
                    n_h = length(idx_t);
                    w_h = zeros(n_h,1);
                    for j = 1:n_h
                        idx_h = obj.paras.MBM.ht(:,i) == idx_t(j);
                        [~,w_h(j)] = normalizeLogWeights(obj.paras.MBM.w(idx_h));
                    end
                    %Recycle
                    temp = obj.paras.MBM.tt{i}(idx);
                    obj.paras.PPP.w = [obj.paras.PPP.w;log([temp.r]')+w_h];
                    obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                end
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Remove Bernoullis
                    obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                    %Update hypothesis table, if a Bernoulli component is
                    %pruned, set its corresponding entry to zero
                    idx = find(idx);
                    for j = 1:length(idx)
                        temp = obj.paras.MBM.ht(:,i);
                        temp(temp==idx(j)) = 0;
                        obj.paras.MBM.ht(:,i) = temp;
                    end
                end
            end
            
            %Remove tracks that contains no valid local hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                %Ensure the algorithm still works when all Bernoullis are
                %recycled
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i) > 0;
                [~,~,obj.paras.MBM.ht(idx,i)] = unique(obj.paras.MBM.ht(idx,i),'rows','stable');
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows','stable');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
            
        end
        
        function obj = PMBM_predict(obj,P_S,motionmodel,birthmodel)
            %PMBM_PREDICT performs PMBM prediction step.
            for obj_idx=1:length(obj.paras.MBM.tt)
                for local_idx=1:length(obj.paras.MBM.tt{obj_idx})
                    obj.paras.MBM.tt{obj_idx}(local_idx) = obj.Bern_predict(obj.paras.MBM.tt{obj_idx}(local_idx),motionmodel,P_S);
                end
            end
            obj = obj.PPP_predict(motionmodel,birthmodel,P_S);
        end
      
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,w_min,M)
            %PMBM_UPDATE performs PMBM update step.
            %INPUT: z: measurements --- array of size (measurement
            %       dimension x number of measurements)
            %       gating: a struct with two fields that specifies gating
            %       parameters: P_G: gating size in decimal --- scalar;
            %                   size: gating size --- scalar.
            %       wmin: hypothesis weight pruning threshold --- scalar in
            %       logarithmic scale
            %       M: maximum global hypotheses kept

            % Initialization
            update_MBM.tt = {};  
            update_MBM.ht = [];  %  0 indicate no assignment to the obj, and length(tt{obj_idx}) + 1 indicate miss detection
            update_MBM.w = [];
            log_lik_MBM = {};
            m_k = size(z, 2);  % Num measurements
            z_to_MBM = zeros(size(z, 2), 1);  % This array map a detection to an object having potential to be a Bernoulli detection
            P_D = sensormodel.P_D;
            dummy_state = struct('x', NaN, 'P', NaN);

            % Perform ellipsoidal gating for each Bernoulli state density and each mixture component in the PPP intensity.
            Bern_in_gate = cell(length(obj.paras.MBM.tt), 1);
            PPP_in_gate = [];
            for obj_idx=1:length(obj.paras.MBM.tt)
                Bern_in_gate{obj_idx} = zeros(length(obj.paras.MBM.tt{obj_idx}), m_k);
                for local_idx=1:length(obj.paras.MBM.tt{obj_idx})
                    [~, meas_in_gate] = obj.density.ellipsoidalGating(obj.paras.MBM.tt{obj_idx}(local_idx).state, z, measmodel, gating.size);
                    Bern_in_gate{obj_idx}(local_idx, :) = meas_in_gate';
                end
            end

            for i=1:length(obj.paras.PPP.w)
                [z_ingates, meas_in_gate] = obj.density.ellipsoidalGating(obj.paras.PPP.states(i), z, measmodel, gating.size);
                PPP_in_gate = cat(1, PPP_in_gate, meas_in_gate');
            end

            % Bernoulli update. For each Bernoulli state density, create a misdetection hypothesis 
            % (Bernoulli component), and m object detection hypothesis (Bernoulli component), 
            % where m is the number of detections inside the ellipsoidal gate of the given state density.
            for obj_idx=1:length(obj.paras.MBM.tt)
                update_MBM.tt{obj_idx} = repmat(struct('r', 0, 'state', dummy_state), length(obj.paras.MBM.tt{obj_idx}), m_k+1);
                log_lik_MBM{obj_idx} = -inf(length(obj.paras.MBM.tt{obj_idx}), m_k+1);
                for local_idx=1:length(obj.paras.MBM.tt{obj_idx})
                    for z_idx=1:m_k
                        if Bern_in_gate{obj_idx}(local_idx, z_idx)
                            lik_detected = obj.Bern_detected_update_lik([obj_idx, local_idx],z(:, z_idx),measmodel,P_D);
                            Bern = obj.Bern_detected_update_state([obj_idx, local_idx],z(:, z_idx),measmodel);
                            update_MBM.tt{obj_idx}(local_idx, z_idx) = Bern;
                            log_lik_MBM{obj_idx}(local_idx, z_idx) =lik_detected;
                        end
                    end
                    [Bern, lik_undetected] = obj.Bern_undetected_update([obj_idx, local_idx],P_D);
                    update_MBM.tt{obj_idx}(local_idx, m_k + 1) = Bern;
                    log_lik_MBM{obj_idx}(local_idx, m_k + 1) = lik_undetected;
                end
            end

            % Update PPP with detections. Note that for detections that are not inside the gate of 
            % undetected objects, the corresponding likelihood is simply the clutter intensity
            
            for z_idx=1:size(z, 2)
                if sum(PPP_in_gate(:, z_idx)) > 0
                    [Bern, lik_new] = obj.PPP_detected_update(PPP_in_gate(:, z_idx),z(:, z_idx),measmodel,P_D,sensormodel.intensity_c);
                    update_MBM.tt{end+1} = [];
                    update_MBM.tt{end} = [update_MBM.tt{end}; Bern];
                    log_lik_MBM{end+1} = [];
                    log_lik_MBM{end} = [log_lik_MBM{end}; lik_new];
                    z_to_MBM(z_idx) = length(update_MBM.tt);
                end
            end
            % For each global hypothesis, construct the corresponding cost matrix and use Murty's 
            % algorithm to obtain the M best global hypothesis with highest weights. Note that for 
            % detections that are only inside the gate of undetected objects, they do not need to 
            % be taken into account when forming the cost matrix.
            hypothesis_num = size(obj.paras.MBM.ht, 1);

            % If there is no global hypothesis, put potential obj to look
            % up table
            if hypothesis_num == 0 || isempty(obj.paras.MBM.ht) || isempty(obj.paras.MBM.w)
                obj.paras.MBM.w = zeros(1, 1);
                obj.paras.MBM.ht = ones(1, 1);
            end

            % If there is global hypothesis
            for hypo_idx=1:max(1, hypothesis_num)
                N_h = length(obj.paras.MBM.tt);  % Num obj
                % Construct loss matrix. Row is measurement and column
                % represent object or potential obj/cluster.
                L = -inf(m_k, N_h+m_k);
                
                % Loop through all obj to assign loss for each obj
                for obj_idx=1:N_h
                    % Get local hypothesis index of an obj
                    local_hypothesis_idx = obj.paras.MBM.ht(hypo_idx, obj_idx);

                    % if local_hypothesis_idx is 0 or global hypothesis do
                    % not assign to the obj, keep loss to 0
                    if local_hypothesis_idx == 0
                        continue
                    end

                    % Get its state and exist prob
                    local_state = obj.paras.MBM.tt{obj_idx}(local_hypothesis_idx).state;
                    local_r = obj.paras.MBM.tt{obj_idx}(local_hypothesis_idx).r;
                    predict_x = measmodel.h(local_state.x);
                    predict_P = measmodel.H(local_state.x) * local_state.P * measmodel.H(local_state.x)' + measmodel.R;

                    % make the matrix positive
                    predict_P = (predict_P + predict_P') / 2;
                
                    % Loop through all measurement
                    for z_idx=1:m_k 
                        % check if measurement in obj gate, construct the loss
                        if Bern_in_gate{obj_idx}(local_hypothesis_idx, z_idx)
                            L(z_idx, obj_idx) = log(local_r) + log(P_D) + log_mvnpdf(z(:, z_idx), predict_x, predict_P) ...
                                - log(1 - local_r + local_r * (1 - P_D));
                        end
                    end
                end
                % Loop through all measurement to assign loss for potential
                % obj/cluster
                for z_idx=1:m_k
                    if sum(PPP_in_gate(:, z_idx)) > 0
                        [~, lik_new] = obj.PPP_detected_update(PPP_in_gate(:, z_idx),z(:, z_idx),measmodel,P_D,sensormodel.intensity_c);
                        L(z_idx, z_idx + N_h) = lik_new;
                    else
                        L(z_idx, z_idx + N_h) = log(sensormodel.intensity_c);
                    end
                end

                % Invert cost matrix
                L = -L;
                
                [col4rowBest,row4colBest,gainBest] = kBest2DAssign(L,M);
                for m=1:min(M, size(col4rowBest, 2))
                    % Update log weight of global hypothesis
                    update_log_w = obj.paras.MBM.w(hypo_idx);
                    update_MBM.ht = [update_MBM.ht; zeros(1, length(update_MBM.tt))];
                    curr_global_idx = size(update_MBM.ht, 1);
                    for z_idx=1:m_k
                        obj_idx = col4rowBest(z_idx, m);
                        % if detection is associate to previous obj
                        if obj_idx <= N_h
                            curr_local_idx = obj.paras.MBM.ht(hypo_idx, obj_idx);
                            update_MBM.ht(curr_global_idx, obj_idx) = (curr_local_idx - 1) * (m_k + 1) + z_idx;
                            update_log_w = update_log_w + log_lik_MBM{obj_idx}(curr_local_idx, z_idx);
                        % if detection is associate to potential obj,
                        % indicate from gating
                        elseif z_to_MBM(z_idx) > 0
                            update_MBM.ht(curr_global_idx, z_to_MBM(z_idx)) = 1;
                            update_log_w = update_log_w + log_lik_MBM{z_to_MBM(z_idx)}(1);
                        % if detection is cluster, update log weight only
                        else
                            update_log_w = update_log_w + log(sensormodel.intensity_c);
                        end
                    end
                    % if object is miss detection, modify the lookup table
                    for obj_idx=1:N_h
                        if update_MBM.ht(curr_global_idx, obj_idx) == 0
                            curr_local_idx = obj.paras.MBM.ht(hypo_idx, obj_idx);

                            % if local_hypothesis_idx is 0 or global hypothesis do
                            % not assign to the obj, do not update
                            if curr_local_idx == 0
                                continue
                            end
                            update_MBM.ht(curr_global_idx, obj_idx) = (curr_local_idx - 1) * (m_k + 1) + m_k + 1;
                            update_log_w = update_log_w + log_lik_MBM{obj_idx}(curr_local_idx, m_k+1);
                        end
                    end
                    update_MBM.w = [update_MBM.w; update_log_w];
                end
            end

            % 'flatten' update_MBM.tt
            for obj_idx=1:N_h
                dummy_tt = update_MBM.tt{obj_idx}';
                update_MBM.tt{obj_idx} = dummy_tt(:)';
            end

            % Normalize log_weights
            update_MBM.w = normalizeLogWeights(update_MBM.w);

            % Update PPP with misdetection:
            obj = obj.PPP_undetected_update(P_D);

            % Reduction
            % Prune and cap global hypotheses
            [update_MBM.w, update_MBM.ht] = hypothesisReduction.prune(update_MBM.w, update_MBM.ht, w_min);
            [update_MBM.w, update_MBM.ht] = hypothesisReduction.cap(update_MBM.w, update_MBM.ht, M);

            % Prune local hypothesis that is not in global
            % hypotheses and reindex lookup table
            for obj_idx=1:size(update_MBM.ht, 2)
                local_idx_in_ht = update_MBM.ht(:, obj_idx);
                idx_in_global_hypothesis = unique(local_idx_in_ht(local_idx_in_ht ~= 0));
                update_MBM.tt{obj_idx} = update_MBM.tt{obj_idx}(idx_in_global_hypothesis);
                dummy_map = zeros(length(local_idx_in_ht) + 1, 1);
                for idx=1:length(idx_in_global_hypothesis)
                    dummy_map(idx_in_global_hypothesis(idx) + 1) = idx;
                end
                update_MBM.ht(:, obj_idx) = dummy_map(local_idx_in_ht + 1);
            end


            % Remove the object that is never assign in any global
            % hypothesis, and remove the obj in MBM.tt
            objs_to_keep = sum(update_MBM.ht, 1) ~= 0;  % Find columns that are not all zeros
            
            % Keep only columns that have at least one non-zero element
            update_MBM.ht = update_MBM.ht(:, objs_to_keep);
            
            % Remove corresponding entries in MBM.tt
            update_MBM.tt = update_MBM.tt(objs_to_keep);

            % Merge similar global hypothesis and update weights
            % Find unique rows in update_MBM.ht and their corresponding indices
            [uniqueKeys, ~, idx] = unique(update_MBM.ht, 'rows');
            
            % Initialize the new weight array
            newWeight = zeros(size(uniqueKeys, 1), 1);
            
            % Sum weights for identical rows
            for i = 1:max(idx)
                [~, newWeight(i)] = normalizeLogWeights(update_MBM.w(idx == i));
            end
            
            % Update the structure with the new unique keys and summed weights
            update_MBM.ht = uniqueKeys;
            update_MBM.w = newWeight;

            obj.paras.MBM = update_MBM;
        end
        
        function estimates = PMBM_estimator(obj,threshold)
            %PMBM_ESTIMATOR performs object state estimation in the PMBM
            %filter
            %INPUT: threshold (if exist): object states are extracted from
            %       Bernoulli components with probability of existence no
            %       less than this threhold in Estimator 1. Given the
            %       probabilities of detection and survival, this threshold
            %       determines the number of consecutive misdetections
            %OUTPUT:estimates: estimated object states in matrix form of
            %       size (object state dimension) x (number of objects)
            %%%
            %First, select the multi-Bernoulli with the highest weight.
            %Second, report the mean of the Bernoulli components whose
            %existence probability is above a threshold. 
            
            [~, best_global_idx] = max(exp(obj.paras.MBM.w));
            estimates = [];
            for obj_idx=1:size(obj.paras.MBM.ht, 2)
                local_idx_in_ht = obj.paras.MBM.ht(best_global_idx, obj_idx);

                % if the local idx of the obj with that hypothesis is 0,
                % there is no obj
                if local_idx_in_ht == 0
                    continue
                end

                % Else we get the state if the prob of exist greater than
                % the threshold
                obj_state_w = obj.paras.MBM.tt{obj_idx}(local_idx_in_ht);
                if obj_state_w.r > threshold
                    estimates = [estimates, obj_state_w.state.x];
                end
            end
        end
    
    end
end