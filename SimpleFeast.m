%
% A SIMPLE ONE LAYER FEAST WITH DIFFERENT NUMBER OF FEATURES
% Author: Saeed Afshar
% Email: S.Afshar[at]westernsydney[dot]edu[dot]au
% A dummy dataset with a square moving around in a circle and observing
% a feast layer with different number of neurons for each run. 

%%
rows       = 13; % height of the input image in pixels
cols       = rows;  % width of the input image in pixels
cntPoint = rows/2; % location of the centre point;
I0       = zeros(rows,cols);  % Image presented to the network  initially all zeros
sumAllImages     = I0;           % Initializing the variable that will hold the summation image of all the input images presented

locationNoisePower = 0.3;  % Jitter in the location of a point source that is moving in our image (in pixels)

ts = 1e6;  % data presentations or time samples let's say in seconds
w0 = 20;   % Rotation frequency of the point source in radian/(timeunits)
for t = 1:ts   % show what the data looks like
    
    % set the current row column location of the point.
    col = round(3*cos(2*pi*t/w0) + cntPoint + randn*locationNoisePower+.5);
    row = round(3*sin(2*pi*t/w0) + cntPoint + randn*locationNoisePower+.5);
    
    I = I0;                       % Initialize input image to zeros
    %I(round(row),round(col)) = 1; % set the point to one on the image
    I((row-1):(row+1),(col-1):(col+1)) = 1; % set the point to one on the image
    sumAllImages = sumAllImages + I;  % keep a summation of all the images generated
    
    if t<(w0*3)     %% show animation of the input image and the average for a little while
        if mod(t,2)==1
            figure(1);
            subplot(2,1,1); imagesc(I); colorbar; axis image; title(['Input Image: Frame = ' num2str(t) ]);
            
            subplot(2,1,2); imagesc(sumAllImages); colorbar;axis image; title('Summed Images')
            drawnow
        end
    end
end
figure(1);  % show the final result of the data
subplot(2,1,1); imagesc(I); colorbar; axis image; title(['Input Image. Frame = ' num2str(t) ]);
subplot(2,1,2); imagesc(sumAllImages); colorbar;axis image; title('Summed Images')
drawnow



%%

for nNeuron = [1 2 4 9 16]  % number of neurons
    displayFreq      = 1000; % in units of time
    nextTimeSample = displayFreq; % Plotting frequency
    eta = 0.001;             % learning rate how much each new input affects the neuron model
    thresholdRise = 0.001;   % how much a hit makes threshold become more selective
    thresholdFall = 0.010;   % how much a miss makes threshold become less selective more open to new inputs
    tau = 2;
    
    sqNeuron = ceil(sqrt(nNeuron));  % For plotting
    
    % Initialize the Matrix for storing the last timestamp of the event at
    % a pixel
    timeStamps = zeros(rows,cols)-inf; 
    
    w = rand(cols*rows,nNeuron); % initialize random neurons
    for iNeuron = 1:nNeuron
        w(:,iNeuron)       = w(:,iNeuron)./norm(w(:,iNeuron));  % normalize neurons
    end
    
    missCount            = 0;
    
    thresh               = zeros(1,nNeuron)+1;  % keep a record of threshold values
    neuronWinCount       = zeros(1,nNeuron);  % keep a record of neuron spike counts
    
    threshRecord         = nan(ts,nNeuron);
    neuronWinCountRecord = nan(ts,nNeuron);
    missCountRecord      = nan(ts,1);
    
    for t = 1:ts
        
        % Generating a square of events by calculating the positions of the
        % square
        col = round(3*cos(2*pi*t/w0) + cntPoint + randn*locationNoisePower+.5);
        row = round(3*sin(2*pi*t/w0) + cntPoint + randn*locationNoisePower+.5);
        
        % Recording the timestamps of the events                        
        timeStamps((row-1):(row+1),(col-1):(col+1)) = t;
        
        % Calculate the time surface of the past events.
        timeSurface = exp((timeStamps-t)/tau);
        timeSurface_Vector    = timeSurface(:)';
        timeSurface_Vector    = timeSurface_Vector/norm(timeSurface_Vector);
        
        % Find the cosine distance of the time surface with each neuron
        dotProds        = timeSurface_Vector*w;
        
        % Find the closest neuron which satisfies the threshold
        [C,winnerNeuron ]       = max(dotProds.*(dotProds > thresh));
        if all(C==0)
            % If none of the dot products crosses the threshold then relax
            % the thresholds of all the neurons
            thresh = thresh - thresholdFall;  % example value of  thresholdFall  is 0.002
            missCount = missCount +1;
        else
            % Update the winner neuron weights
            w(:,winnerNeuron)           = (1-eta)*w(:,winnerNeuron) + eta*(timeSurface(:));  % example value of  eta  is 0.001
            w(:,winnerNeuron)           = w(:,winnerNeuron)./norm(w(:,winnerNeuron));
            
            % Increase the threshold of the winner neuron
            thresh(winnerNeuron)   = thresh(winnerNeuron) + thresholdRise;     % example value of  thresholdFall  is 0.001
            neuronWinCount(winnerNeuron) = neuronWinCount(winnerNeuron) + 1;
            
        end
        threshRecord(t,:)         = thresh;
        neuronWinCountRecord(t,:) = neuronWinCount;
        missCountRecord(t)        = missCount;
        
        
        if t > nextTimeSample  % show the simulation every once in a while
            
            nextTimeSample = nextTimeSample + displayFreq;
            displayFreq = displayFreq*1.1;
            figure(4);
            wSummed = I0;
            
            % Plot the neurons
            for iNeuron = 1:nNeuron
                subplot(sqNeuron,sqNeuron,iNeuron)
                wShow = reshape(w(:,iNeuron),cols,rows);
                wSummed = wSummed + wShow;
                imagesc(wShow);
                %title([ num2str(thresh(iNeuron),2)   '-'  num2str(neuronWinCount(iNeuron))])
                set(gca,'visible','off')
                set(findall(gca, 'type', 'text'), 'visible', 'on')
                
            end
            figure(5)
            imagesc(wSummed);% colorbar;
            title('The weights of all the neurons summed together')
            drawnow
            
            figure(6)
            subplot(3,1,1);
            plot(threshRecord(1:t,:));set(gca,'xscale','log');grid on;title('Threshold values')
            
            subplot(3,1,2);
            plot(movmean(diff(neuronWinCountRecord(1:t,:)),1000));set(gca,'xscale','log');grid on;title('win rate for each neuron')
            
            subplot(3,1,3);
            plot(movmean(diff(missCountRecord(1:t)),1000));set(gca,'xscale','log');grid on;title('network miss rate')
            
            
        end
    end
end
    
    
    
    
    
    
    
    
    
    
