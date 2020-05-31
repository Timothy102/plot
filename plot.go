//Package plot implements plotting functionality for all sorts of data as well as matrix and vector visualisations. It also provides statistical components of datasets, such as variances and covariances.
package plot

import (
	"fmt"
	"image/color"
	"log"
	"os"

	"github.com/timothy102/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

//DrawApproximation draws a line of approximation based on k,n which should be the outputs of ApproximiateLine
func DrawApproximation(p *plot.Plot, points []point, k, n float64, file string) error {
	max := FindMaxX(points)
	l, err := plotter.NewLine(plotter.XYs{
		{0, n}, {max, max*k + n},
	})
	if err != nil {
		return fmt.Errorf("could not create line :%v", err)
	}
	p.Add(l)
	return nil

}

//PlotGaussian plots the Gaussian distribution with the default filename of "Gaussian"
func PlotGaussian(mean, stddev float64, iterations int) error {
	var points []point
	iter := (mean*2 - (-mean * 2)) / float64(iterations)
	for i := -mean * 2; i < mean*2; i += iter {
		p := point{
			x: i,
			y: Gaussian(i, mean, stddev),
		}
		points = append(points, p)
	}
	if err := PlotPoints(points, "Gaussian.png", true); err != nil {
		return fmt.Errorf("could not plot the Gaussian :%v", err)
	}
	return nil
}

//PlotPoints plots the points onto the file. By default it shows the standard deviation on the graph.
func PlotPoints(points []point, file string, showStddev bool) error {
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	defer f.Close()
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}

	xys := PointToXYs(points)
	s, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.Color = color.RGBA{B: 255, A: 255}
	p.Add(s)
	if showStddev {
		if err := DrawStddev(p, points); err != nil {
			return fmt.Errorf("could not draw std dev of the dataset :%v", err)
		}
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
func createPlot(file string) (*plot.Plot, error) {
	f, err := os.Create(file)
	if err != nil {
		return nil, fmt.Errorf("could not create file %s:%v", file, err)
	}
	defer f.Close()
	p, err := plot.New()
	if err != nil {
		return nil, fmt.Errorf("could not create plot")
	}
	wt, err := p.WriterTo(500, 500, "png")
	if err != nil {
		return nil, fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return nil, fmt.Errorf("could not write to %s :%v", file, err)
	}
	return p, nil
}

//PlotMatrix returns the input matrix plotted to a file.
func PlotMatrix(m matrix.Matrix, file string) error {
	p, err := createPlot(file)
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	if m.NumberOfRows() > 2 || m.NumberOfColumns() > 2 {
		return fmt.Errorf("Matrix you're trying to plot should be a 2x2 matrix.")
	}

	var points []point
	ps := RandomPoints(10, -1.0, 1.0)

	vecs := PointToVector(ps)
	for _, v := range vecs {
		if v.GetLength() == 0 {
			log.Println("discarding bat data point")
			continue
		}
		if v.GetLength() == 1 {
			p := point{
				x: v.Slice()[0],
				y: v.Slice()[1],
			}
			points = append(points, p)
		}

	}
	xys := PointToXYs(points)

	//scatter for the original matrix vectors
	s, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter: %v", err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.Color = color.RGBA{R: 255, A: 255}
	l, err := plotter.NewLine(xys)
	if err != nil {
		return fmt.Errorf("could not draw line :%v", err)
	}
	p.Add(s)
	p.Add(l)

	vectors := PointToVector(points)
	var transformed []matrix.Vector
	for _, v := range vectors {
		transformed = append(transformed, v.ApplyMatrix(m))
	}
	resultXYS := VectorToXYs(transformed)

	//scatter for the transformed matrix vectors,
	sc, err := plotter.NewScatter(resultXYS)
	if err != nil {
		return fmt.Errorf("could not create scatter: %v", err)
	}
	sc.GlyphStyle.Shape = draw.CrossGlyph{}
	sc.Color = color.RGBA{B: 255, A: 255}

	l2, err := plotter.NewLine(resultXYS)
	if err != nil {
		return fmt.Errorf("could not draw line :%v", err)
	}
	p.Add(sc)
	p.Add(l2)

	return nil
}

//PlotVector plots the vector onto file.
func PlotVector(v matrix.Vector, file string) error {
	p, err := createPlot(file)
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	/*if v.NumberOfColumns() != 2 || v.NumberOfRows() != 1 {
		return fmt.Errorf("cannot plot anything other than a 2x1 vector.")
	}
	*/
	l, err := plotter.NewLine(plotter.XYs{
		{0, 0},
		{v.Slice()[0], v.Slice()[1]},
	})
	if err != nil {
		return fmt.Errorf("could not create line :%v", err)
	}
	l.Color = color.RGBA{R: 255, A: 255}
	l.Width = 1.0

	p.Add(l)
	return nil
}

func PlotVectors(vectors []matrix.Vector, file string) error {
	p, err := createPlot(file)
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	for _, vec := range vectors {
		/*
			if vec.NumberOfColumns() != 2 || vec.NumberOfRows() != 2 {
				return fmt.Errorf("cannot plot anything other than a 2x2 vector.")
			}
		*/
		l, err := plotter.NewScatter(plotter.XYs{
			{0, 0},
			{vec.Slice()[0], vec.Slice()[1]},
		})

		if err != nil {
			return fmt.Errorf("could not create line :%v", err)
		}
		p.Add(l)
	}

	return nil
}
