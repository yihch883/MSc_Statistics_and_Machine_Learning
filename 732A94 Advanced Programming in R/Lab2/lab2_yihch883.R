rm(list=ls())
name <- "YiHung Chen"
liuid <-"yihch883"

#1,1,1
sheldon_game<-function(player1,player2){
  
  elements1<-list("rock","paper","scissors","lizard","spock")
  if(player1 %in% elements1 & player2 %in% elements1){
    if(player1 == player2){return("Draw!")}
    else if(player1=="rock" & (player2=="lizard"||player2=="paper")){return("Player 1 wins!")}
    else if(player1=="paper" & (player2=="spock"||player2=="rock")){return("Player 1 wins!")}
    else if(player1=="scissors" & (player2=="paper"||player2=="lizard")){return("Player 1 wins!")}
    else if(player1=="lizard" & (player2=="spock"||player2=="paper")){return("Player 1 wins!")}
    else if(player1=="spock" & (player2=="rock"||player2=="scissors")){return("Player 1 wins!")}
  
    else if(player2=="rock" & (player1=="lizard"||player1=="paper")){return("Player 2 wins!")}
    else if(player2=="paper" & (player1=="spock"||player1=="rock")){return("Player 2 wins!")}
    else if(player2=="scissors" & (player1=="paper"||player1=="lizard")){return("Player 2 wins!")}
    else if(player2=="lizard" & (player1=="spock"||player1=="paper")){return("Player 2 wins!")}
    else if(player2=="spock" & (player1=="rock"||player1=="scissors")){return("Player 2 wins!")}
  }
  else {stop()}
  
  
  }


#1.2.1
my_moving_median<-function(x,n,...){
  if (is.vector(x)==TRUE & is.numeric(n)==TRUE){
    outputlist<-c()
    maxnum <- length(x)-n
    for(i in 1:maxnum){
      range <-x[i:(i+n)]
      check <- NA %in% range
    if( check == FALSE){
      movemedian <- median(range)
      outputlist[i] <- movemedian
    }
    
    else if(check == TRUE || ...==TRUE){
      movemedian <- median(range,...)
      outputlist[i] <- movemedian
    }
    else if(check == TRUE & ...==TRUE){
      
      outputlist[i] <-NA
    }
    }
    return(outputlist)
  }
  else{stop()}
  
  
}


#1.2.2
for_mult_table <- function(from, to){
  if (is.numeric(from)==TRUE & is.numeric(to)==TRUE){
   

    numbers <-c(from:to)
    tabledata <-matrix(data=NA, nrow=length(numbers),ncol=length(numbers))
   
    for(i in 1:length(numbers)){
      for(j in 1:length(numbers)){
        tabledata[i,j]<- numbers[i]*numbers[j]
         }
      
    }
    rownames(tabledata)<-(numbers)
    colnames(tabledata)<-(numbers)
    return(tabledata)
  }
  else{stop()}

}

#1.3.1
find_cumsum <- function(x, find_sum){
if (is.numeric(x)==TRUE & is.numeric(find_sum)==TRUE){
  
  i<- 1
  sum <- 0
  while(i <= length(x) & sum < find_sum){
    sum <-sum+x[i]
    i<- i+1
  }
  
  return(sum)
}
else{stop()}
}


#1.3.2
while_mult_table<-function(from, to){
    if (is.numeric(from)==TRUE & is.numeric(to)==TRUE){
      
      
      numbers <-c(from:to)
      tabledata <-matrix(data=NA, nrow=length(numbers),ncol=length(numbers))
      
      i<-1
      j<-1
      while(i<=length(numbers)){
        while(j<=length(numbers)){
          tabledata[i,j]<- numbers[i]*numbers[j]
          j<-j+1
          
        }
        j<-1        #reset j to make sure we start on the right column.
        i<-i +1
        
      }
      rownames(tabledata)<-(numbers)
      colnames(tabledata)<-(numbers)
      return(tabledata)
      
    }
    else{stop()}
    
  }

#1,4,1
repeat_find_cumsum<-function(x,find_sum){
    if (is.numeric(x)==TRUE & is.numeric(find_sum)==TRUE){
      
      i<- 1
      sum <- 0
      repeat{
        sum <-sum+x[i]
    
        i<- i+1
      
      if(i > length(x) | sum > find_sum){break}
        }
      
      return(sum)
    }
    else{stop()}
  }

#1.4.2
repeat_my_moving_median<-function(x,n,...){
  if (is.vector(x)==TRUE & is.numeric(n)==TRUE){
    outputlist<-c()
    maxnum <- length(x)-n
    i <-1
    repeat{
      range <-x[i:(i+n)]
      check <- NA %in% range
      if( check == FALSE){
        movemedian <- median(range)
        outputlist[i] <- movemedian
        i <-i+1
        if(i > maxnum)(break)
      }
      
      else if(check == TRUE || ...==TRUE){
        movemedian <- median(range,...)
        outputlist[i] <- movemedian
        i <-i+1
        if(i > maxnum)(break)
      }
      else if(check == TRUE & ...==TRUE){
        
        outputlist[i] <-NA
        i <-i+1
        if(i > maxnum)(break)
      }
      
    }
    
    return(outputlist)
  }
  else{stop()}
  
  
}

#1.5.1
in_environment <- function(env){
    
    env <- search()[length(search())]
    outcome1<-ls(env)
    return(outcome1)
  }
  
#1.6.1
cov<-function(X){
  if (is.data.frame(X)==TRUE){
  outcome2 <- sapply(X,FUN=function(X){sd(X)/mean(X)})
  return(outcome2)
  
  }
  else{stop()}
}

#1.7.1  
moment<-function(i){
  if(is.numeric(i)==TRUE){
  function(X){
    return(sum((X-mean(X))^i)/length(X)) #equation according to wiki
  }
  }
  else{stop()}
}


