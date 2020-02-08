%  create a channel map file

Nchannels = 15; % number of channels
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;

xcoords = [11.0, 59.0, 27.0, 43.0, 11.0, 59.0, 27.0, 43.0, 11.0, 59.0, 27.0, 43.0, 11.0, 59.0, 27.0];
ycoords = [20.0, 40.0, 40.0, 60.0, 60.0, 80.0, 80.0, 100.0, 100.0, 120.0, 120.0, 140.0, 140.0, 160.0, 160.0];
kcoords   = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

fs = 30000.0; % sampling frequency
save(fullfile('chanMap.mat'), ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
