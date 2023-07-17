#' Djikstra Algorithm
#' 
#' Takes a graph and an initial node, then calculates the shortest path from the initial node to every other node in the graph.
#' 
#' @param data A data.frame with three variables (v1, v2 and w) that contains the nodes of the graph (from v1 to v2) with the length of the node (w)
#' @param initial_node The node to start with.
#' @usage dijkstra(data, initial_node)
#' @return The shortest path from the initial node to every other node.
#' @export 
#' @examples
#' dijkstra(wiki_graph, 1)
#' dijkstra(wiki_graph, 3)
#' @references \url{https://en.wikipedia.org/wiki/Dijkstra\%27s_algorithm}


dijkstra <-function(data, initial_node){
  
    pick_the_smallest_value <- function(subvector, i){
        sorted_subvector <- sort(subvector)
        i_smallest_value <- sorted_subvector[i]
        return(which(i_smallest_value == subvector))
    }  
  
   # Update the pathdata dataframe when we find new shortest path  
   update_distance <- function(indexes_of_neighbours, distances, visited, data, pathdata, previous_node, index_first_of_current_node){
       for(index in indexes_of_neighbours){
           if(data[index, 2] %in% visited){
               next
           }
           else{
               new_distance  <- data[index, 3] + pathdata[data[index_first_of_current_node, 1], 4]
               current_distance <- pathdata[data[index, 2], 4]
               if ( new_distance < current_distance){
                   pathdata[data[index, 2], 4] <- new_distance
                   pathdata[data[index, 2],3] <- data[index_first_of_current_node, 1]
               }
           }
       }
      return(pathdata)
   }
  
   #----------------------CHECK IF INPUTS ARE CORRECT----------------------------- 
  
   if(length(data[,1]) != length(data[,2]) | length(data[,1]) != length(data[,3])){stop()}
   if(any(colnames(data) != c("v1", "v2", "w"))){stop()} # Check if the names of each column of dataframe are correct
   if(is.data.frame(data)==TRUE & initial_node %in% data[,1] ){
    
    #----------------------CREATE THE TABLE CALLED PATHDATA-----------------------------
      allnodes<-data[,1]  # Take every elements from first column of input data
      unvisited<-allnodes[!duplicated(allnodes)] # Find out how many different elements in first column of data
      visited<-c() # Create a empty vector to store visited nodes later on
      first_column<-rep(initial_node,length(unvisited)) # Create The first collumn of the table that we make 
      path_length<-rep(Inf,length(unvisited))
      previous_nodes<-rep(NA,length(unvisited))
      everynodes<-allnodes[!duplicated(allnodes)] # Use to create path dataframe. In order not to confused with unvisited vector.
      previous_node <- initial_node
      pathdata<-data.frame(first_column,everynodes,previous_nodes,path_length) #create a path dataframe to make calculation easier to see
      #----------------------CREATE THE TABLE CALLED PATHDATA-----------------------------
      
      
      #---------------- FOLLOWING THE DIJKSTRA ALGORITHM FOR THE INITIAL NODE ------------
      # Find the index of nodes which are next to initial_node
      indexes_of_neighbours<- which(allnodes %in% initial_node) # Find the row index of initial_node
      node_nextto_init <-data[indexes_of_neighbours,2] # Find the node next to initial_node
    
      # Step1: set the path length to initial_node=0
      init_index_ineverynodes<- which(everynodes %in% initial_node)
      pathdata[init_index_ineverynodes, 4] <- 0
      pathdata[init_index_ineverynodes, 3] <- initial_node 
      visited <-append(visited,initial_node) # Moved the initial node to the visited nodes
    
      # Step2: calculate the path length next to initial_node
      length_of_nodes_next_to_init <- data[indexes_of_neighbours,3] # Get the length data from data(wiki_graph)
      index_neighbors_next_to_current_node<- which(data[,1]==initial_node) # Find the row index of initial_node
      nodes_nextto_current_node <-data[indexes_of_neighbours,2] # Find the node next to initial_node
      index_first_of_current_node <- indexes_of_neighbours[1]
      index_closest_node_to_current_node <- index_first_of_current_node + which(data[index_neighbors_next_to_current_node,3] == min(data[index_neighbors_next_to_current_node,3])) -1 # The variable contains the number of the closest node
      distances = data[indexes_of_neighbours, 2]
      closest_node = data[index_closest_node_to_current_node, 2]
      closest_node_index_in_pathdata<- which(everynodes %in% closest_node) #find the index of the closet node in pathdata
      pathdata[closest_node_index_in_pathdata,4]<- data[index_closest_node_to_current_node,3] #update the length
      pathdata[closest_node_index_in_pathdata,3]<- initial_node # update the previous node
      previous_node <- closest_node
      #---------------- FOLLOWING THE DIJKSTRA ALGORITHM FOR THE INITIAL NODE ------------
      

      #---------------- MOVE TO THE CLOSEST NODE OF INITIAL NODE -------------------------      
      unvisited <- unvisited[-initial_node] # Removing the initial node from the unvisited vector
      unvisited <- unvisited[- which( closest_node == unvisited)]
      pathdata <- update_distance(indexes_of_neighbours, distances, visited, data, pathdata,  
                                  previous_node, index_first_of_current_node)
      #---------------- MOVE TO THE CLOSEST NODE OF INITIAL NODE -------------------------      
    while(length(unvisited) > 0 ){
      indexes_of_neighbours<- which(allnodes %in% closest_node) # The indexes of neighbor nodes in the given dataframe.
      index_neighbors_next_to_current_node<- which(data[,1]==closest_node)
      nodes_nextto_current_node <-data[index_neighbors_next_to_current_node,2] # Find the node next to current node
      index_first_of_current_node <- indexes_of_neighbours[1]
      index_closest_node_to_current_node <- index_first_of_current_node + which(data[index_neighbors_next_to_current_node,3] == min(data[index_neighbors_next_to_current_node,3])) -1 # The variable contains the number of the closest node
      distances = data[indexes_of_neighbours, 2]
      previous_node <- data[index_first_of_current_node, 1]
      closest_node = data[index_closest_node_to_current_node, 2]
      
      i<-1

      repeat{
        
        stop <- FALSE
        check<- (!(closest_node %in% visited)) #aka if the closest node it unvisited.  It returns just one element

        
        for (boolean_value in check){
          if (boolean_value == FALSE){
            stop<-TRUE
          }
        }
        
        # If the closest node is already visited, we find the 2nd closest node, etc.
        if (stop){
          
          index_closest_node_to_current_node <- index_first_of_current_node + pick_the_smallest_value(data[index_neighbors_next_to_current_node, 3], i) -1 # The variable contains the number of the closest node
          distances = data[indexes_of_neighbours, 2]
          previous_node <- data[index_first_of_current_node, 1]
          closest_node = data[index_closest_node_to_current_node, 2]
          i<-i+1
          next
        }
        pathdata <- update_distance(indexes_of_neighbours, distances, visited, data, pathdata, previous_node, index_first_of_current_node)
        
        i <- 1
        break
      }
      
      visited <-append(visited,closest_node)
      closest_node_index_in_pathdata<- which(everynodes %in% closest_node) #find the index of the closet node in pathdata
      pathdata[closest_node_index_in_pathdata,3]<- previous_node # update the previous node
      unvisited <- unvisited[- which( closest_node == unvisited)]
    }
    
    return(pathdata[, 4])    

  }
  else{stop()}
  
}