# Learning with Kernels: Numerical Experiment
*AMS 747 final project* **Yifei Chen, Hongyuan Fu, Rex Wang**

This numerical experiment illustrates the effect of kernel smoothness on the kernel's ability to approximate smooth and non-smooth functions. We generated a simple example with three kernels of various smoothness fitting two different target functions in Python following the numerical experiment in Sec. 7.7 of "Learning Theory from First Principles" by Francis Bach. 

## Setup
### Target functions
The target functions are a smooth sinusoid and a non-smooth square wave: 
```math
\begin{align*}
    f_\text{smooth}(x) &= \sin (4\pi x)\\
    f_\text{non-smooth} &= \begin{cases}
        1 & x \mod {0.5} < 0.25\\
        -1 & \text{otherwise}
    \end{cases}
\end{align*}
```

### Data generation
Input $x_i$ are randomly and independently drawn from the uniform distribution $\mathcal{U}(0,1)$. Then, the target values $y_i = f_\*(x_i) + \varepsilon_i$, where $f_\*$ is either the smooth or non-smooth target function, and $\varepsilon_i \sim \mathcal{N}(0, 0.25^2)$ is the random noise. 

### Kernels 
From the "roughest" to the "smoothest", the kernels are:
* exponential kernel
```math
    k_\text{exponential}(x,x') = \exp\left(-\frac{\|x - x'\|_2}{r}\right)
```
* Matérn kernel of order 3
```math
        k_\text{Mat\'{e}rn}(x, x') = \left(1 + \sqrt{5}\frac{\|x - x'\|_2}{r} + \frac{5}{3}\frac{\|x - x'\|_2^2}{r^2}\right)\exp\left(-\frac{\|x - x'\|_2}{r}\right)
```
* Gaussian kernel
```math
        k_\text{Gaussian}(x, x') = \exp\left(-\frac{\|x - x'\|_2^2}{r^2}\right)
```

In these formulae, $r$ is the "kernel bandwidth" (with unit homogeneous to $x$) and is chosen to be the median of all pairwise distances $\|x_i - x_j\|_2$ of the training data (Bach) 

### Experiment
For each sample size $n$ chosen from the 15 values logarithmically spaced between 10 and 1000, $(x_i, y_i)^n_{i=1}$ were generated for the smooth and non-smooth functions each as described above. Then, models with regularization parameter $\lambda = \frac{k(x_i,x_i)^2}{n}$ (Bach) were trained using kernel ridge regression with the three different kernels. Then, the excess risk was estimated over 1000 evenly spaced $\hat{x}_i$ using the formula 
```math
    \text{excess risk} \approx \frac{1}{1000}\sum^{1000}_{i=1}[\hat{f}(\hat{x}_i) - f_*(\hat{x}_i)]^2
```
where $\hat{f}$ is the trained model. Finally, this step was repeated twenty times and the average excess risk for each $n$ was recorded.
