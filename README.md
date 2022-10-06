# When installing

If you run into an error along the lines `could not find library libmetis-gtsam.so`, return to the GTSAM build folder and do
```
sudo make install
sudo ldconfig
``` 
(probably best solution) or add
```
export LD_LIBRARY_PATH=/usr/local/lib/:$HOME/local/lib/:$LD_LIBRARY_PATH
```
to your `~/.bashrc`/`~/.zshrc` or potentially
```
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib/:$HOME/local/lib/:$LD_LIBRARY_PATH' >> ~/.zshrc
```
to your Dockerfile.