mex_dir = fullfile(pwd, '/build_mex');
srcs = {'mex_lbp.cpp'};
srcs = cellfun(@(s) fullfile(mex_dir, s), srcs,'UniformOutput', false);
include_dir = fullfile(pwd, '/include');
mex('CXXFLAGS=$CXXFLAGS -std=c++17', srcs{:}, ['-I', include_dir]);
