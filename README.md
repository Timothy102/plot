# plot
Go simple-to-use library for plotting data, including the Gaussian distribution and Taylor series approximation as well as statistical computations of datasets.

You can upload your own dataset to plot or use a dataset with the correspondant output. Including the sigmoid and gaussian distribution.

[![GoDoc](https://godoc.org/github.com/Timothy102/plot?status.svg)](https://godoc.org/github.com/Timothy102/plot)

The plot package is very easy to use, all the plotting is done for you with Gonum.
Looking forward to get some feedback!

Let's take a look at some functionality!

  #Installation
  ```
  go get github.com/timothy102/plot
  ```

Let's look at a dataset using the cosine function. Second and third argument to the function invoke the starting and end point of iteration. You can adjust the iterations parameter to get a more dense distribution.
```
points:=DefineDataset(math.Cos,-1.0,1.0,1000)
if err:=PlotPoints(points,"Graph.png",false);err!=nil{
  log.Fatalf("could not plot data :%v",err)
}
```

That is how simple it is. If you would like to import your dataset externally, use the ReadFromDatafile function.
The rest is the same.

```
points,err:=ReadFromDatafile(filepath)
//error handling
if err:=PlotPoints(points,"Graph.png",false);err!=nil{
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

