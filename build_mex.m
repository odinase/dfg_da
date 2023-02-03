RELEASE = true;
DEBUG = false & ~RELEASE;



if RELEASE
    flags = '-O3 -DNDEBUG';
elseif DEBUG
    flags = '-g';
else
    flags = '';
end



build_dir = '/build_mex';
mex_dir = fullfile(pwd, build_dir);
srcs = {fullfile(mex_dir, 'lbp_mex.cpp'),...
    './src/hypothesis.cpp',...
    './src/lbp.cpp'
};
eigen_path = '/usr/include/eigen3'; % Have no idea how to get around this...
includes = {'./include',...
            mex_dir,...
            eigen_path...
};
includes = cellfun(@(s) ['-I', s], includes,'UniformOutput', false);
mex(['CXXFLAGS=$CXXFLAGS', ' ', flags, ' ', '-std=c++17'], srcs{:}, includes{:});
