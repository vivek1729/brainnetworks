library(ggplot2)
library(ggsci)
library(dplyr)
read.csv("linear/allComb_out.csv", header = F) %>% 
    rename(
        size.of.graph = V1,
        size.of.source = V2,
        size.of.sink   = V3,
        density.of.graph = V4,
        size.of.trauma = V5,
        cost.before    = V6,
        cost.after     = V7
    )
