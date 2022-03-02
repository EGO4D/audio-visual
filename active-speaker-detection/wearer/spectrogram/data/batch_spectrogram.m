% Usage: octave-cli batch_spectrogram.m 

for n = 0 : 10000
   n	
   wname = sprintf('../../../vad/audios/%d.wav', n);
   if ~exist(wname)
      continue
   end      
   [x, Fs] = audioread(wname); # audio file
   step = fix(5*Fs/1000);     # one spectral slice every 5 ms
   window = fix(40*Fs/1000);  # 40 ms data window
   fftn = 2^nextpow2(window); # next highest power of 2
   [S, f, t] = specgram(x, fftn, Fs, window, window-step);
   S = abs(S(2:fftn*4000/Fs,:)); # magnitude in range 0<f<=4000 Hz.
   S = log(S+1e-20);
   cmd = sprintf('save spectrograms/%d.mat S f t -mat-binary', n);
   eval(cmd);
end   
