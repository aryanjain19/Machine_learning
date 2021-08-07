function submit()
  addpath('./lib');

  conf.assignmentKey = 'xHfBJWXxTdKXrUG7dHTQ3g';
  conf.itemName = 'Support Vector Machines';
  conf.partArrays = { ...
    { ...
      'drOLk', ...
      { 'gaussianKernel.m' }, ...
      'Gaussian Kernel', ...
    }, ...
    { ...
      'JYt9Q', ...
      { 'dataset3Params.m' }, ...
      'Parameters (C, sigma) for Dataset 3', ...
    }, ...
    { ...
      'UHwLk', ...
      { 'processEmail.m' }, ...
      'Email Preprocessing', ...
    }, ...
    { ...
      'RIiFh', ...
      { 'emailFeatures.m' }, ...
      'Email Feature Extraction', ...
    }, ...
  };
  conf.output = @output;

  submitWithConfiguration(conf);
end

function out = output(partId, auxstring)
  % Random Test Cases
  x1 = sin(1:10)';
  x2 = cos(1:10)';
  ec = 'the quick brown fox jumped over the lazy dog';
  wi = 1 + abs(round(x1 * 1863));
  wi = [wi ; wi];
  if partId == 'drOLk'
    sim = gaussianKernel(x1, x2, 2);
    out = sprintf('%0.5f ', sim);
  elseif partId == 'JYt9Q'
    load('ex6data3.mat');
    [C, sigma] = dataset3Params(X, y, Xval, yval);
    out = sprintf('%0.5f ', C);
    out = [out sprintf('%0.5f ', sigma)];
  elseif partId == 'UHwLk'
    word_indices = processEmail(ec);
    out = sprintf('%d ', word_indices);
  elseif partId == 'RIiFh'
    x = emailFeatures(wi);
    out = sprintf('%d ', x);
  end 
end
