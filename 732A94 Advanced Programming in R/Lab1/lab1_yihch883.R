rm(list=ls())
name <- "YiHung Chen"
liuid <- "yihch883"

#1.1.1
my_num_vector<-function(){
  vector1<- c(log10(11), cos(pi/5),exp(pi/3),(1173%%7/19))
  options(digits=22) #set to max decimals
  return(vector1)
  }


#1.1.2
filter_my_vector<-function(x, leq){
  
     vector2 <- replace(x,x[]>=leq,NA)
     return(vector2)
     
   }

#1.1.3
dot_prod<- function(a,b){
  

  dotprod<-( a%*%b)
  
  ans<-drop(dotprod) #since %*% will give matrix, drop 1 dimension
  return(ans)  
}


#1.1.4
approx_e <- function(N){
  vector3 <-0:N
  approxe <- 1/factorial(vector3)
  approxesum<- sum(approxe)
  
  return(approxesum)
}
  
#1.2.1
my_magic_matrix <- function(){
  
  matrix1<- cbind(c(4,3,8),c(9,5,1),c(2,7,6))
  return(matrix1)
}

#1.2.2
calculate_elements <- function(A){
   elementscount <- nrow(A)*ncol(A)
   return(elementscount)
}

#1.2.3
row_to_zero<- function(A,i){
   zero <-rep(0,ncol(A))
   A[i,] <- zero
   
   return(A)
}

#1.2.4
add_elements_to_matrix<- function(A, x, i, j){
  
  A[i,j]<-(A[i,j]+x)
  return(A)

}
#1.3.1
my_magic_list<- function(){
  matrix4<- cbind(c(4,3,8),c(9,5,1),c(2,7,6))
  magiclist <-list(info="my own list",c(1.04139,0.80902,2.84965,0.21053
), matrix4)
  return(magiclist)
  
}

#1.3.2
change_info <- function(x,text){
  x$info<-text
  newlist1<-x
  return(newlist1)
  
}

#1.3.3
add_note<- function(x, note){
  addlist<-list(note= note)
  newlist2<-append(x,addlist)
  return(newlist2)
  
}
#1.3.4
sum_numeric_parts<-function(x){
  vector4<-unlist(x)
  nochar <-as.numeric(vector4)
  nochar[is.na(nochar)]<-0
  sum<- sum(nochar)
  return(sum)
}

#1.4.1
my_data.frame<-function(){
  id<-c(1,2,3)
  names<-c("John","Lisa","Azra")
  income<-c(7.30,0.00,15.21)
  rich<-c(FALSE, FALSE , TRUE)
  df1 <-data.frame(id,name=names,income,rich)
  
  return(df1)
}

#1.4.2

sort_head<-function(df, var.name,n){
  
  sorting<-order(df[[var.name]],decreasing = TRUE)
  df2<-df[sorting,]

  return(df2[1:n,])

}

#1.4.3
add_median_variable <- function(df, j){
  median <-median(df[,j])
  compared_to_median<-df[,j]-median
   compared_to_median[compared_to_median > 0 ]<-"Greater"
   compared_to_median[compared_to_median == 0 ]<-"Median"
   compared_to_median[compared_to_median < 0 ]<-"Smaller"

  df3<-cbind(df, compared_to_median)

return(df3)
}

#1.4.4
analyze_columns<- function(df, j){
 
 element1<-c(mean=mean(df[,j[1]]),median=median(df[,j[1]]),sd=sd(df[,j[1]]))
 element2<-c(mean=mean(df[,j[2]]),median=median(df[,j[2]]),sd=sd(df[,j[2]]))
 corelation<-cor(cbind(df[,j[1]],df[,j[2]]))
 newlist3<-list(element1,element2,corelation)
 names(newlist3) <- c(names(df)[j[1]], names(df)[j[2]], "correlation_matrix")
 return(newlist3)
 }


