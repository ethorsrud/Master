try
    % prepare for kilosort execution
    addpath(genpath('/home/eirith/Kilosort2'));

    % set file path
    fpath = '/home/eirith/Master/tmp_KS2';

    % add npy-matlab functions (copied in the output folder)
    addpath(genpath(fpath));

    % create channel map file
    run(fullfile('/home/eirith/Master/tmp_KS2/kilosort2_channelmap.m'));

    % Run the configuration file, it builds the structure of options (ops)
    run(fullfile('/home/eirith/Master/tmp_KS2/kilosort2_config.m'))

    ops.trange = [0 Inf]; % time range to sort

    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);

    % time-reordering as a function of drift
    rez = clusterSingleBatches(rez);

    % main tracking and template matching algorithm
    rez = learnAndSolve8b(rez);

    % final merges
    rez = find_merges(rez, 1);

    % final splits by SVD
    rez = splitAllClusters(rez, 1);

    % final splits by amplitudes
    rez = splitAllClusters(rez, 0);

    % decide on cutoff
    rez = set_cutoff(rez);

    fprintf('found %d good units \n', sum(rez.good>0))

    fprintf('Saving results to Phy  \n')
    rezToPhy(rez, fullfile(fpath));
catch
    fprintf('----------------------------------------');
    fprintf(lasterr());
    quit(1);
end
quit(0);
