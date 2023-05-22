different.k <- function(own.sequence, df.train, df.test, name1, name2){
  for (i in own.sequence) {
    mod.knn <- knn.model(df.train, i) # knn.model is another function
    plot(eval_model(mod.knn, df.train, df.test, paste(name1, "(knn: k=",i,")"),
                                                paste(name2, "(knn: k=", i,")")))
  }
  return(mod.knn)
}


different.k.1 <- function(own.sequence, df.train, df.test, name1, name2){
  plot.vec <- NULL
  for (i in own.sequence) {
    mod.knn <- knn.model(df.train,i)
    plot.vec <- c(plot.vec, eval_model(mod.knn, df.train, df.test, name1, name2))
  }
  out.plot <- cowplot::plot_grid(plot.vec, ncol = 2)
  return(out.plot)
}
