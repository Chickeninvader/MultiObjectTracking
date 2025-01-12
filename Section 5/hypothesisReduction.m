classdef hypothesisReduction
    %HYPOTHESISREDUCTION is class containing different hypotheses reduction
    %method 
    %PRUNE: prune hypotheses with small weights.
    %CAP:   keep M hypotheses with the highest weights and discard the rest. 
    %MERGE: merge similar hypotheses in the sense of small Mahalanobis
    %       distance.
    
    methods (Static)
        function [hypothesesWeight, multiHypotheses] = ...
                prune(hypothesesWeight, multiHypotheses, threshold)
            %PRUNE prunes hypotheses with small weights
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: hypotheses with weights smaller than this
            %       threshold will be discarded --- scalar in logarithmic scale
            %OUTPUT:hypothesesWeight: hypotheses weights after pruning in
            %       logarithmic scale --- (number of hypotheses after
            %       pruning) x 1 vector   
            %       multiHypotheses: (number of hypotheses after pruning) x
            %       1 structure  
            dummy_weights = [];
            dummy_hypotheses = [];
            num_hypothesis = size(hypothesesWeight, 1);
            for i=1:num_hypothesis
                if hypothesesWeight(i) > threshold
                    dummy_weights = [dummy_weights; hypothesesWeight(i)];
                    dummy_hypotheses = [dummy_hypotheses; multiHypotheses(i,:)];
                end
            end
            hypothesesWeight = dummy_weights;
            multiHypotheses = dummy_hypotheses;
        end
        
        function [hypothesesWeight, multiHypotheses] = ...
                cap(hypothesesWeight, multiHypotheses, M)
            %CAP keeps M hypotheses with the highest weights and discard
            %the rest 
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector  
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       M: only keep M hypotheses --- scalar
            %OUTPUT:hypothesesWeight: hypotheses weights after capping in
            %       logarithmic scale --- (number of hypotheses after
            %       capping) x 1 vector 
            %       multiHypotheses: (number of hypotheses after capping) x
            %       1 structure 
            % Check the number of hypotheses
            numHypotheses = length(hypothesesWeight);
            
            if M >= numHypotheses
                % If M is greater than or equal to the number of hypotheses, no capping needed
                return;
            end
            
            % Sort the hypotheses by their weights in descending order
            [sortedWeights, sortIdx] = sort(hypothesesWeight, 'descend');
            
            % Keep only the top M hypotheses and their corresponding weights
            topMIdx = sortIdx(1:M);
            hypothesesWeight = sortedWeights(1:M);
            hypothesesWeight = hypothesesWeight / sum(hypothesesWeight, 1);
            multiHypotheses = multiHypotheses(topMIdx);
            
        end
        
        function [hypothesesWeight,multiHypotheses] = ...
                merge(hypothesesWeight,multiHypotheses,threshold,density)
            %MERGE merges hypotheses within small Mahalanobis distance
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector  
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: merging threshold --- scalar
            %       density: a class handle
            %OUTPUT:hypothesesWeight: hypotheses weights after merging in
            %       logarithmic scale --- (number of hypotheses after
            %       merging) x 1 vector  
            %       multiHypotheses: (number of hypotheses after merging) x
            %       1 structure 
            
            [hypothesesWeight,multiHypotheses] = ...
                density.mixtureReduction(hypothesesWeight,multiHypotheses,threshold);
            
        end
    end
end