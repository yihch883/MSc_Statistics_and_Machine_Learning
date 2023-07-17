#'  The Lab5 API package
#' 
#'  Using OpenStreetMaps via Stamen Tiles combine with crime and wind data from ggmap.
#'  
#' @param inputId inputId
#' @param label label
#' @param value value
#' @param ... other input
#'
#' @import ggplot2
#' @import ggmap
#' @import dplyr 
#' @importFrom  shiny fluidPage
#' @export textInputRow
textInputRow<-function (inputId, label, value = "", ...) 
{
  shiny::div(style="display:inline-block",
      shiny::tags$label(label, `for` = inputId), 
      shiny::tags$input(id = inputId, type = "text", value = value,class="input-small", ...))
}


#================= FRIND CRIME FROM TYPE =================

#' find_crime_from_type
#'
#' @param data crim_data
#' @param type crime_type
#' @return crime_map
#' @export
find_crime_from_type <- function(data,type){
  
  lon<-NULL
  lat<-NULL
  
  if(type != "theft" & 
     type != "auto theft" & 
     type != "murder"& 
     type != "robbery"& 
     type != "aggravated assault"& 
     type != "burglary"){stop("Wrong Input, only: theft, auto theft, murder,robbery,aggravated assault,burglary.can be searched")}
  
  
  index_of_type <- which(data$offense == as.character(type))
  data_of_type <- data[index_of_type,]
  
  huston <- c(left= -95.4, bottom = 29.6, right = -95, top = 29.8)
  map <- ggmap::get_stamenmap(huston, maptype = "terrain", zoom=10)
  crime_map<- ggmap::ggmap(map) + ggplot2::geom_jitter(data = data_of_type, ggplot2::aes(x= lon, y= lat),size = 0.005, color = "red")
  
  return(crime_map)
}

#================= FIND CRIME FROM MONTH =================

#' find_crime_from_month
#'
#' @param data crime_data
#' @param time month input
#' @return crime_map
#' @export
find_crime_from_month<-function(data,time){
  lon<-NULL
  lat<-NULL
  
  if(time == 1){month <- "january"}
  else if(time == 2){month <- "february"}
  else if(time == 3){month <- "march"}
  else if(time == 4){month <- "april"}
  else if(time == 5){month <- "may"}
  else if(time == 6){month <- "june"}
  else if(time == 7){month <- "july"}
  else if(time == 8){month <- "august"}
  else if(time > 8 | time <= 0 ){stop("Month should be an integer from 1 to 8")}
  
  
  index_of_month <- which(data$month == month)
  data_of_the_month <- data[index_of_month,]
  
  huston <- c(left= -95.4, bottom = 29.6, right = -95, top = 29.8)
  map <-ggmap::get_stamenmap(huston, maptype = "terrain", zoom=10)
  crime_map <-ggmap(map) + ggplot2::geom_jitter(data = data_of_the_month, ggplot2::aes(x= lon, y= lat),size = 0.0005, color = "red")
  
  return(crime_map)
}


#============ FIND CRIME FROM TIME AND TYPE =============

#' find_crime_from_time_and_type
#'
#' @param data crime_data
#' @param time month input
#' @param type crime_type
#' @return crime_map
#' @export
find_crime_from_time_and_type <- function(data,type,time){
  
  lon<-NULL
  lat<-NULL
  
  ###Check type input###
  if(type != "theft" & 
     type != "auto theft" & 
     type != "murder"& 
     type != "robbery"& 
     type != "aggravated assault"& 
     type != "burglary"){stop("Wrong Input, only: theft, auto theft, murder,robbery,aggravated assault,burglary.can be searched")}
  
  ###Check month input###
  if(time == 1){month <- "january"}
  else if(time == 2){month <- "february"}
  else if(time == 3){month <- "march"}
  else if(time == 4){month <- "april"}
  else if(time == 5){month <- "may"}
  else if(time == 6){month <- "june"}
  else if(time == 7){month <- "july"}
  else if(time == 8){month <- "august"}
  else if(time > 8 | time <= 0 ){stop("Month should be an integer from 1 to 8")}
  
  index_of_month <- which(data$month == month)
  data_of_the_month <- data[index_of_month,]
  
  data <- ggmap::crime
  
  index_of_type <- which( data$offense == type)
  data_of_type <- data_of_the_month[index_of_type,]
  
  huston <- c(left= -95.4, bottom = 29.6, right = -95, top = 29.8)
  map <- ggmap::get_stamenmap(huston, maptype = "terrain", zoom=10)
  crime_map <- ggmap(map) + ggplot2::geom_jitter(data = data_of_type, ggplot2::aes(x= lon, y= lat),size = 0.005, color = "red")
  
  return(crime_map)
}

#=========== CREATE A NEW TABLE FOR WIND DATA ===========

updated_wind <- ggmap::wind %>% rowwise() %>% 
  mutate(lon2 = lon + delta_lon) %>%
  mutate(lat2 = lat + delta_lat)

#== FUNCTION FOR CREATING THE MAP RAGARDING THE INPUTS===

#' wind_plot
#'
#' @param min min 
#' @param max max
#'
#' @return plot1
#' @export
wind_plot <- function(min, max){
  lat<-lat2<-lon<-lon2<-spd<-NULL
  if(min > max){
    stop("Min is bigger than Max")
  }
  
  else if((min %% 1 != 0) | ((max %% 1) != 0)){
    stop("Should be integer")
  }
  else{
  updated_wind <- updated_wind %>%
    filter(
      -95.4 <= lon &  lon <= -95,
      29.6 <=  lat & lat <= 29.8,
      min <=  spd &  spd <= max
    )
  huston <- c(left= -95.4, bottom = 29.6, right = -95, top = 29.8)
  map <-get_stamenmap(huston, maptype = "terrain", zoom=10)
  plot1 <- ggmap(map) + geom_segment(data = updated_wind, aes(x= lon, y= lat, xend= lon2, yend= lat2, color= spd), alpha=0.8, size=2) 
  plot(plot1)
  
  }
}
