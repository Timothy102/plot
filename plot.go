package plot

import (
	"fmt"
	"image/color"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

//PlotGaussian plots the Gaussian distribution with the default filename of "Gaussian"
func PlotGaussian(mean, stddev float64, iterations int) error {
	var points []point
	for i := 0.0; i < 2.0; i += 1 / float64(iterations) {
		p := point{
			x: i,
			y: Gaussian(i, mean, stddev),
		}
		points = append(points, p)
	}
	if err := PlotPoints("Gaussian.png", points); err != nil {
		return fmt.Errorf("could not plot the Gaussian :%v", err)
	}
	return nil
}

//PlotPoints plots the points onto the file. By default it shows the standard deviation on the graph.
func PlotPoints(file string, points []point) error {
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	defer f.Close()
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	xys := make(plotter.XYs, len(points))
	for i, p := range points {
		xys[i].X = p.x
		xys[i].Y = p.y
	}
	s, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.Color = color.RGBA{B: 255, A: 255}
	p.Add(s)
	if err := DrawStddev(p, points); err != nil {
		return fmt.Errorf("could not draw std dev of the dataset :%v", err)
	}
	wt, err := p.WriterTo(500, 500, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("could not write to %s :%v", file, err)
	}
	return nil
}

//DrawStddev plots the standard deviation
func DrawStddev(pl *plot.Plot, points []point) error {
	l, err := plotter.NewLine(plotter.XYs{
		{ComputeAverageX(points) - ComputeStddevX(points), ComputeAverageY(points)},
		{ComputeAverageX(points) + ComputeStddevX(points), ComputeAverageY(points)},
	})
	if err != nil {
		return fmt.Errorf("could not draw line :%v", err)
	}
	pl.Add(l)
	return nil
}
