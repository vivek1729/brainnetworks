library(ggplot2)
library(ggsci)
library(dplyr)
source("~/Box/R/template.r")
readdf <- function(filename){
read.csv(filename, header = F) %>% 
    rename(
        size.of.graph = V1,
        size.of.source = V2,
        size.of.sink   = V3,
        density.of.graph = V4,
        size.of.trauma = V5,
        max.flow.before = V6,
        max.flow.after = V7,
        cost.before    = V8,
        cost.after     = V9,
        cost.time = V10
    ) ->df
    return(df)
}

df <- readdf("linear/allComb_out.csv")
df %>% 
    summary

# Tue Dec  4 12:24:46 2018 ------------------------------
#' size cost
df %>% 
    filter(size.of.source == 16) %>% 
    filter(size.of.sink == 16) %>% 
    filter(size.of.trauma == 32) %>%
    mutate(
        # x = size.of.trauma,
        # y = (cost.after-cost.before) / (max.flow.before - max.flow.after)
        x = size.of.graph,
        y = cost.time,
        group = naturalsort::naturalfactor(density.of.graph)
    ) %>% 
    plot_simple_points2(
        'number of nodes',
        'time cost',
        'density of graph'
    )+
    geom_line(aes(color = group)) +
    scale_color_npg2() +
    theme(legend.position = c(0.2,0.8)) +
    plot_save_pub('costtime-per-numberofnodes')

#sizeofgraphset2 Tue Dec  4 13:11:15 2018 ------------------------------
df <- readdf('sizeofgraphset2/allComb_out.csv')

df %>% 
    summary

# Tue Dec  4 12:24:46 2018 ------------------------------
#' size cost
df %>% 
    filter(size.of.graph < 1000) %>% 
    # filter(size.of.source == 16) %>% 
    # filter(size.of.sink == 16) %>% 
    # filter(size.of.trauma == 32) %>%
    mutate(
        # x = size.of.trauma,
        # y = (cost.after-cost.before) / (max.flow.before - max.flow.after)
        x = size.of.graph,
        y = cost.time,
        group = naturalsort::naturalfactor(density.of.graph)
    ) %>% 
    plot_simple_points2(
        'number of nodes',
        'time cost',
        'density of graph'
    )+
    geom_line(aes(color = group)) +
    scale_color_npg2() +
    theme(legend.position = c(0.2,0.8)) +
    plot_save_pub('costtime-per-numberofnodes')

