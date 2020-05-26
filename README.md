# plot
Go simple-to-use library for plotting data as well as statistical computations on datasets, such as variances and covariances.

You can upload your own dataset to plot or use a dataset with the correspondant output. Including the sigmoid and gaussian distribution.

You can find the full documentation on GoDoc: godoc.org/timothy102/plot .


The plot package is very easy to use, all the plotting is done for you with Gonum.
Looking forward to get some feedback!

Let's take a look at some functionality!


Let's look at a predefined dataset using the cosine function.
The arguments -1.0 and 1.0 are defined as the starting and end point of iteration. Their values are due to the fact that cosine varies from -1 to 1. You can adjust the iterations parameter(which is set to 10000) to get a more dense distribution.
```
points:=DefineDataset(math.Cos,10000,-1.0,1.0)
if err:=PlotPoints("Graph.png",points);err!=nil{
  log.Fatalf("could not plot data :%v",err)
}
```

That is how simple it is. If you would like to import your dataset externally, use the ReadFromDatafile function.
The rest is the same.

```
points,err:=ReadFromDatafile(filepath)
//error handling
if err:=PlotPoints("Graph.png",points);err!=nil{
  log.Fatalf("could not plot data :%v",err)
}
```


And finally, the Gaussian distribution.
First parameter is the mean and second is the standard deviation of the dataset you would like to graph.
```
if err:=PlotGaussian(1.0,2.0); err!=nil{
  log.Fatalf("could not plot data :%v",err)
}
```

![Gaussian](C:\Users\cvetk\Documents\gocode\matrix\plot\Gaussian.png)
