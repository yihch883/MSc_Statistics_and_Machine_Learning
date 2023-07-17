library(ggplot2)
set.seed(12345)

#Question 1a
s <- 22
n <- 70
alpha_0 <- 8
beta_0 <- 8
f <- n-s

alpha <- alpha_0+s
beta <- beta_0+f
real_mean <- alpha/(alpha+beta)
real_sd <- sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1)))

mean <- c()
sd <- c()

for(i in seq(0,10000)){
  nDraws <- rbeta(i,alpha,beta)
  mean <- append(mean,mean(nDraws))
  sd <- append(sd,sd(nDraws))
}

plot(mean,type = 'l')
lines(x=seq(0,10000),y=rep(real_mean,10001),col="red")

plot(sd,type = 'l')
lines(x=seq(0,10000),y=rep(real_sd,10001),col="red")

#Question 1b
real_prob <- pbeta(0.3,alpha,beta,lower.tail = FALSE)
nDraws <- rbeta(10000,alpha,beta)
sample_prob <- sum(nDraws>0.3)/length(nDraws)

cat("The difference bettween real and simulation probability are",sample_prob-real_prob,"which is very close")

#Question 1c
hist(nDraws/(1-nDraws),breaks = 100)
plot(density(nDraws/(1-nDraws)))

#-----------------------------------

#Question 2a
obs <- c(33,24,48,32,55,74,23,17)
n <- length(obs)-1

calculate_tau <- function(mu)
{
  res <- (sum((log(obs) - mu)^2))/n
  return(res)
}

tau_2 <- calculate_tau(mu = 3.6)

X <- rchisq(10000,df = n)
xs <- (n*tau_2)/X

xs_df <- as.data.frame(xs)

ggplot(data = xs_df, aes(x = xs)) + 
  geom_histogram(aes(y = ..density..), color = "darkblue", fill = "lightblue",bins = 100) +
  labs(title = "Histogram of Posterior Distribution", x = "X", y = "Count") 

#Question 2b
phi_z <- sqrt(xs)/sqrt(2)
G <- (2 * pnorm(phi_z,mean = 0,sd = 1)) -1

G_df <- as.data.frame(G)

ggplot(data = G_df, aes(x = G)) + 
  geom_histogram(aes(y = ..density..), color = "darkblue", fill = "lightblue",bins = 100) +
  labs(title = "Histogram of Posterior Distribution of GINI Coefficient", x = "X", y = "Count")

#Question 2c
lower_b <- quantile(G,0.025)
upper_b <- quantile(G,0.975)

CI <- c(lower_b,upper_b)
CI

#Question 2d
kdens_estimate <- density(G)

dens_df <- data.frame(x = kdens_estimate$x,y  = kdens_estimate$y)

ordered_indices <- order(dens_df$y,decreasing = TRUE)

ordered_dens_df <- dens_df[ordered_indices,]

ordered_dens_df$csum <- cumsum(ordered_dens_df$y)

cutoff <- 0.95* ordered_dens_df$csum[dim(ordered_dens_df)[1]]

HPdensity <- ordered_dens_df[ordered_dens_df$csum <= cutoff,]

HPDIntervals <- c(min(HPdensity$x),max(HPdensity$x))

ggplot(data = G_df, aes(x = G)) + 
  geom_histogram(aes(y = ..density..), color = "darkblue", fill = "darkblue",bins = 100) +
  labs(title = "Histogram of Posterior Distribution of GINI Coefficient", x = "X", y = "Count") +
  geom_segment(aes(x = CI[1],y = 0.5,yend =  0.5,xend = CI[2],colour = 'Equal tail'))  +
  geom_segment(aes(x = HPDIntervals[1],y = 0.7,yend =  0.7,xend = HPDIntervals[2],colour = 'HPDI tail'))

#--------------------------------
#Question 3a
posterior_func_before_normal <- function(k,data,lambda,mu){
  data <- c( -2.79, 2.33, 1.83, -2.44, 2.23, 2.33, 2.07, 2.02, 2.14, 2.54)
  lambda <- 0.5
  mu <- 2.4
  n <- length(data)
  elem1 <- (1/(2*pi*besselI(k,nu=0)))**n
  elem2 <- sum(cos(data-mu))-lambda
  result <- lambda*elem1*exp(k*elem2)
  
  
  return (result)
}

k <- seq(0,10,0.001)
integration_factor=integrate(posterior_func_before_normal, lower =0 , upper = 10)[[1]]

posterior_func_norm <- function(k){
  data <- c( -2.79, 2.33, 1.83, -2.44, 2.23, 2.33, 2.07, 2.02, 2.14, 2.54)
  lambda <- 0.5
  mu <- 2.4
  n <- length(data)
  elem1 <- (1/(2*pi*besselI(k,nu=0)))**n
  elem2 <- sum(cos(data-mu))-lambda
  result <- lambda*elem1*exp(k*elem2)
  
  
  return (result/integration_factor)
}
testintegrate= integrate(posterior_func_norm, lower =0 , upper = 10)[[1]]
cat("The integration of normalized posterior distribution is ",testintegrate)
res_vec <- posterior_func_norm(k)
plotdf <- data.frame(k,res_vec)

ggplot(plotdf)+geom_line(aes(x=k,y=res_vec))

#Question 3b
max_index <- which.max(plotdf$res_vec)
post_mode <- plotdf[max_index,1]

cat("The posterior mode of k is ",post_mode )