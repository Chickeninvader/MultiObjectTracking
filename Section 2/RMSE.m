function root_mean_square_error = RMSE(state_sequence1, state_sequence2)
    % RMSE calculates the root mean square error between state_sequence1 and
    % state_sequence2.
    % INPUT: state_sequence1: a cell array of size (total tracking time x 1),
    %        each cell contains an object state mean vector of size
    %        (state dimension x 1)
    %        state_sequence2: a cell array of size (total tracking time x 1),
    %        each cell contains an object state mean vector of size
    %        (state dimension x 1)
    % OUTPUT: root_mean_square_error: root mean square error --- scalar

    % Ensure both state sequences are the same length
    assert(length(state_sequence1) == length(state_sequence2), ...
        'The state sequences must have the same length.');
    
    % Initialize the sum of squared errors
    sum_squared_error = 0;
    total_tracking_time = length(state_sequence1);
    
    % Loop over each time step
    for t = 1:total_tracking_time
        % Extract the state vectors from the cell arrays
        state1 = state_sequence1{t};
        state2 = state_sequence2{t};
        
        % Ensure both state vectors are the same dimension
        assert(length(state1) == length(state2), ...
            'State vectors must have the same dimension.');
        
        % Calculate the squared difference and accumulate
        squared_error = (state1 - state2).^2;
        sum_squared_error = sum_squared_error + sum(squared_error);
    end
    
    % Calculate the mean squared error
    mean_squared_error = sum_squared_error / total_tracking_time;
    
    % Take the square root to obtain RMSE
    root_mean_square_error = sqrt(mean_squared_error);
end