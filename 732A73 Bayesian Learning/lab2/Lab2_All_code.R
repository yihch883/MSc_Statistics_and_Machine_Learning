# Q1 

# 1a) 
library(mvtnorm)
library(readxl)
library(ggplot2)
#read dataset 

temperature_data <- read_xlsx('Linkoping2022.xlsx')
temperature_data$datetime <- as.Date(temperature_data$datetime)

create_time <- function(x)
{
  res <- as.numeric(x - as.Date('2022-01-01')) / 365
  
  res
}

temperature_data$time <- create_time(temperature_data$datetime)
temperature_data$time2 <- temperature_data$time^2

time_mat <- as.matrix(cbind("bias" = 1,"time" = temperature_data$time,"time_2" = (temperature_data$time)^2))
# lecture 5 
# get sigma^2 from inv chi-sq simlator using v_0 & sigma_0^2

#Question 2a

n <- dim(temperature_data)[1]-1
InvChiSq <- function(sample_size,n,tau2)
{
  X <- rchisq(sample_size,df = n)
  xs <- (n*tau2)/X
  return(xs)
}

# prior var 
prior_var <- function(v0,s2)
{
  pvar <- InvChiSq(sample_size = 1,n = v0,tau2 = s2)
  return(pvar)
}

# now get beta given variance from mvt norm distribution
# joint prior 
prior_beta <- function(mu0,sigma_2,omega0_inv){
  betaprior <- rmvnorm(1,mean = mu0, sigma = sigma0_2*omega0_inv)
  return(betaprior)
}

# initialize starting hyperparameters
sigma0_2 <- 1
v_0 <- 1
mu0 <- matrix(c(0,100,-100),nrow = 3,ncol = 1)
omega0_inv <- solve(diag(x = 0.01,nrow = 3,ncol = 3))

prior_draws <- c()
sigma_2 <- prior_var(v0 = 1,s2 = 1)
plot_df <- list()
set.seed(123)
p <- ggplot() #+ geom_point(x = 'time',y = 'temp',data = temperature_data)
for(i in 1:10){
  val <- prior_beta(mu0 = mu0,sigma_2 = sigma_2,omega0_inv = omega0_inv)
  prior_draws <- c(prior_draws,val)
  y <- time_mat %*% t(val)
  #print(val)
  #print(y)
  plot_df[[i]] <- data.frame(cbind("x" = temperature_data$time,"y" = y))
  p <- p + geom_line(data = plot_df[[i]], aes(x = x, y = V2), col = "blue")
  
}
p + geom_point(data = temperature_data,aes(x = time,y = temp))

# rerun with changed hyperparameters 
v0 <- 4
sigma0_2 <- 12
mu0 <- matrix(c(-6,75,-75),nrow = 3,ncol = 1)
omega0_inv <- solve(diag(x = 0.6,nrow = 3,ncol = 3))

# different results for every run,since seed is different
prior_draws <- c()
sigma_2 <- prior_var(v0 = v0,s2 = sigma0_2) 
plot_df <- list()
p <- ggplot() 
for(i in 1:20){
  val <- prior_beta(mu0 = mu0,sigma_2 = sigma_2,omega0_inv = omega0_inv)
  prior_draws <- c(prior_draws,val)
  y <- time_mat %*% t(val)
  plot_df[[i]] <- data.frame(cbind("x" = temperature_data$time,"y" = y))
  p <- p + geom_line(data = plot_df[[i]], aes(x = x, y = V2), col = "blue")+ 
    theme_bw()
}
p + geom_point(data = temperature_data,aes(x = time,y = temp)) +
  labs(x = "Time",y= "Temperature",title = "Prior Curves with Optimal Hyperparameters ")+
  theme(plot.title = element_text(hjust = 0.5))


# compute posterior 
v_n <- v_0 + n
omega_n <- (t(time_mat)%*%time_mat) + solve(omega0_inv)
#lecture slide 4
beta_hat <- solve(t(time_mat)%*%time_mat) %*% t(time_mat) %*% temperature_data$temp

mu_n <- (solve((t(time_mat)%*%time_mat) + solve(omega0_inv))) %*%
  (((t(time_mat)%*%time_mat)%*%beta_hat) + solve(omega0_inv)%*%mu0)

v_n_sigma2_n <- v_0*sigma0_2 + ((t(temperature_data$temp)%*%temperature_data$temp) +
                                  (t(mu0)%*%solve(omega0_inv)%*%mu0 )- (t(mu_n)%*%omega_n%*%mu_n))
v_n_sigma2_n/v_n

# poster variance 
posterior_var <- function(v_n,sigma_2)
{
  pvar <- InvChiSq(sample_size = 1,n = v_n,tau2 = sigma_2)
  return(pvar)
}

# posterior betas
posterior_beta <- function(mu_n,sigma_2,omega_n){
  betaposterior<- rmvnorm(1,mean = mu_n, sigma = as.vector(sigma_2)*solve(omega_n))
  return(betaposterior)
}



posterior_draws <- c()
sigma_2 <- v_n_sigma2_n/v_n
posterior_var(v_n = v_n,sigma_2 = sigma_2)
posterior_beta(mu_n = mu_n,sigma_2 = sigma_2,omega_n = omega_n)

## plotting posterior
plot_df <- list()
p <- ggplot() 
for(i in 1:10){
  val <- posterior_beta(mu_n = mu_n,sigma_2 = sigma_2,omega_n = omega_n)
  posterior_draws <- c(posterior_draws,val)
  y <- time_mat %*% t(val)
  #print(val)
  #print(y)
  plot_df[[i]] <- data.frame(cbind("x" = temperature_data$time,"y" = y))
  p <- p + geom_line(data = plot_df[[i]], aes(x = x, y = V2), col = "blue")
  
}
p + geom_point(data = temperature_data,aes(x = time,y = temp))

## sample from posterior, then plot marginal densities 
posterior_mat <- matrix(nrow = 10000,ncol = 3)
posterior_variance <- c()
for(i in 1:10000){
  posterior_mat[i,] <- posterior_beta(mu_n = mu_n,sigma_2 = sigma_2,omega_n = omega_n)
  posterior_variance[i] <- posterior_var(v_n = v_n,sigma_2 = sigma_2)
  
}

colnames(posterior_mat) <- c('b0','b1','b2')

# posterior plots for b0,b1 and b2
hist(posterior_mat[,1],breaks = 100)
hist(posterior_mat[,2],breaks = 100)
hist(posterior_mat[,3],breaks = 100)

# posterior plot for sigma2
hist(posterior_variance,breaks = 100)

median(posterior_mat[,1])


f_time  <- time_mat%*%t(posterior_mat)

stat_df <- data.frame('lower' = 0,'median' = 0,'upper' = 0)

for(i in seq(0,dim(f_time)[1])){
  #print(i)
  lower <- quantile(f_time[i,],probs = 0.05)
  med <- median(f_time[i,])
  upper <- quantile(f_time[i,],probs = 0.95)
  stat_df[i,] <- c(lower,med,upper)
}

combined_df <- data.frame(cbind(temperature_data,stat_df))


p <- ggplot() + geom_point(data = combined_df,aes(x = time,y = temp))+
  geom_line(data =  combined_df,aes(x = time, y = lower, col = "lower CI "))+
  geom_line(data =  combined_df,aes(x = time, y = median, col = "median"))+
  geom_line(data =  combined_df,aes(x = time, y = upper, col = "upper CI"))+ theme_bw()

p

#1c) 
# y = b0 + b1*x + b2 * x^2
# diff wrt x to find maximal point,solve for x.

maxX <- -(posterior_mat[,2])/(2 * posterior_mat[,3])
ggplot()+ geom_histogram(data = as.data.frame(maxX),aes(x = maxX),bins = 100)
hist(maxX)


#1d)

#To avoid overfitting the data, we use a regularization prior.


#--------------------------------------------------------------------------
                          ### Assignment 2 ###
#--------------------------------------------------------------------------

rm(list=ls())

library("mvtnorm")
library("ggplot2")

WomenAtWork <- read.delim("WomenAtWork.dat", header = TRUE, sep="") 
glmModel<- glm(Work ~ 0 + ., data = WomenAtWork, family = binomial)
summary(glmModel)

women_df <- WomenAtWork[,2:ncol(WomenAtWork)]
lable <- WomenAtWork[, 1]

Npar <- dim(women_df)[2]

# Initialize prior
mu <- as.matrix(rep(0, Npar))
tau <- 2
Sigma <- tau^2 * diag(Npar) #tau^2I

LogPostLogistic <- function(betas,y,X,mu,Sigma){
  X = as.matrix(X)
  linPred <- X%*%betas
  logLik <- sum( linPred*y - log(1 + exp(linPred)) )
  logPrior <- dmvnorm(betas, mu, Sigma, log=TRUE)
  return(logLik + logPrior)
}

# Initialize betas
initVal <- matrix(0, Npar, 1)


# Optimizer
OptimRes <- optim (initVal, LogPostLogistic, gr = NULL, y = lable, X = women_df, mu = mu, Sigma = Sigma, method=c("BFGS"), control=list(fnscale=-1), hessian=TRUE)

beta_mode <- OptimRes$par

# Printing the results to the screen
names(OptimRes$par) <- colnames(women_df) # Naming the coefficient by women_df
approxPostStd <- sqrt(diag(solve(-OptimRes$hessian))) # Computing approximate standard deviations.
names(approxPostStd) <- colnames(women_df)# Naming the coefficient by women_df
print('The posterior mode is:')
print(OptimRes$par)
print('The approximate posterior standard deviation is:')
print(approxPostStd)

upper<-beta_mode+1.96*approxPostStd
lower<-beta_mode-1.96*approxPostStd
cat("The intervals for the variable NSmallChild are upper=",upper[6],"lower=",lower[6])
print(-OptimRes$hessian)




#####Q2-2

x <- c(1,18,11,7,40,1,1)
sigma = solve(-OptimRes$hessian) #since the input of rmvnorm is covariance not sd

sim_draw <- function(x,mean,sigma){
  #convert x 
  x <- as.matrix(x)
  beta <- rmvnorm(n=1, mean = mean, sigma =sigma)
  
  #logistic regression
  elem1 <- exp(t(x)%*%t(beta)) #the transpose of beta is to make sure the dimension correct
  draw <- 1-(elem1/(1 + elem1)) #1- ,Since we are looking for not-working and the equation given is for working
  
  return(draw)
}

pred <- replicate(10000, sim_draw(x,beta_mode,sigma))

plotdf <- as.data.frame(pred)

ggplot(data = plotdf)+geom_histogram(aes(x = pred),bins = 100)


#####Q2-3
thirteen <- replicate(10000,rbinom(1,13,sample(pred)))
thirteen_df <- as.data.frame(thirteen)
ggplot(data = thirteen_df)+geom_histogram(aes(x = thirteen),bins = 13)





