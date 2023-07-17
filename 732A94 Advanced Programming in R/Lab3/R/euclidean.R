#' Euclidean Algorithm
#' 
#' Find the greatest common divisor of two numbers. 
#' 
#' @param a A number.
#' @param b A number.
#' @usage euclidean(a,b)
#' @return greatest common divisor of 'a' & 'b'.
#' @export 
#' @examples
#' euclidean(123612, 13892347912)
#' euclidean(100, 1000)
#' @references \url{https://en.wikipedia.org/wiki/Euclidean_algorithm}

euclidean <- function(a, b){
  
  biggest <- b
  smallest <- a
  
  if (a > b ){
    biggest <- a
    smallest <- b
  }
  if(a<0){
    a <- -a
  }
  
  if(b<0){
    b <- -b
  }
  
  reminder = biggest %% smallest
  remainders_list<- c(a, b, reminder)
  i <- 1
  while( remainders_list[i] > 0) {
   
   remainders_list[i+2] <- remainders_list[i] %% remainders_list[i+1]
   i <- i + 1
      
  }
  
  return(remainders_list[i-1])
}
