//Package plot implements plotting functionality for all kinds of data, including the Gaussian distribution, Taylor series approximations and more. It provides an insightful way to data representation and interpretation along with vector and matrix visualisations.
package plot

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"strings"

	"github.com/timothy102/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

var e = ExponentialEstimate(1, 15)

//PlotsineAndCosine plots all four derivates.
func PlotsineAndCosine(file string) error {
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file :%v", err)
	}
	defer f.Close()
	pts := DefineDataset(math.Sin, -5.0, 5.0, 200)
	xys := PointToXYs(pts)
	sc, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	sc.GlyphStyle.Shape = draw.CrossGlyph{}
	sc.Color = color.RGBA{B: 255, A: 255}

	ps := DefineDataset(math.Cos, -5.0, 5.0, 500)
	xs := PointToXYs(ps)
	s, err := plotter.NewScatter(xs)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.Color = color.RGBA{R: 255, A: 255}

	pts = DefineDataset(math.Sin, -5.0, 5.0, 500)
	pts = FlipOverYAxis(pts)
	xys = PointToXYs(pts)
	sc2, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	sc2.GlyphStyle.Shape = draw.CrossGlyph{}
	sc2.Color = color.RGBA{B: 255, A: 255}

	ps = DefineDataset(math.Cos, -5.0, 5.0, 500)
	ps = FlipOverYAxis(ps)

	xs = PointToXYs(ps)
	s3, err := plotter.NewScatter(xs)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	s3.GlyphStyle.Shape = draw.CrossGlyph{}
	s3.Color = color.RGBA{R: 255, A: 255}

	p.Add(sc, s, sc2, s3)

	wt, err := p.WriterTo(400, 400, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("could not write to  file :%v", err)
	}
	return nil
}

//PlotPoints plots the Points onto the file. By default it shows the standard deviation on the graph.
func PlotPoints(pts Points, file string, showStddev bool) error {
	if !strings.HasSuffix(file, ".png") {
		return fmt.Errorf("File should be a png file :%s", file)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	defer f.Close()
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	ps := DiscludeNan(pts)
	xys := PointToXYs(ps)

	s, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.Color = color.RGBA{B: 255, A: 255}
	p.Add(s)
	if showStddev {
		if err := DrawStddev(p, ps); err != nil {
			return fmt.Errorf("could not draw std dev of the dataset :%v", err)
		}
	}
	wt, err := p.WriterTo(360, 360, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("could not write to %s :%v", file, err)
	}
	return nil
}

//PlotGaussian plots the Gaussian distribution with the default filename of "Gaussian"
func PlotGaussian(mean, stddev float64, iterations int) error {
	var pts Points
	s := mean + stddev
	iter := (s - (mean - stddev)) / float64(iterations)
	for i := (mean - stddev) - mean; i < s+mean; i += iter {
		p := Point{
			X: i,
			Y: Gaussian(i, mean, stddev),
		}
		pts = append(pts, p)
	}
	if err := PlotPoints(pts, "Gaussian.png", true); err != nil {
		return fmt.Errorf("could not plot the Gaussian :%v", err)
	}
	return nil
}

//PlotMatrix returns the input matrix plotted to a file.
func PlotMatrix(m matrix.Matrix, file string) error {
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	if m.NumberOfRows() > 2 || m.NumberOfColumns() > 2 {
		return fmt.Errorf("Matrix you're trying to plot should be a 2x2 matrix")
	}
	var pts Points
	ps := DefineRandomPoints(10, -1.0, 1.0)

	vecs := PointToVector(ps)
	for _, v := range vecs {
		if v.GetLength() == 0 {
			log.Println("discarding bat data Point")
			continue
		}
		if v.GetLength() == 1 {
			p := Point{
				X: v.Slice()[0],
				Y: v.Slice()[1],
			}
			pts = append(pts, p)
		}
	}
	xys := PointToXYs(pts)

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

	vectors := PointToVector(pts)
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
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	if len(v.Slice()) != 2 {
		return fmt.Errorf("cannot plot anything other than a 2x1 vector")
	}
	l, err := plotter.NewLine(plotter.XYs{
		{0, 0},
		{v.Slice()[0], v.Slice()[1]},
	})
	if err != nil {
		return fmt.Errorf("could not create line :%v", err)
	}
	l.Color = color.RGBA{B: 255, A: 255}
	p.Add(l)
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

//PlotVectors plots vectors
func PlotVectors(vectors []matrix.Vector, file string) error {
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	for _, vec := range vectors {
		s := vec.Slice()
		if len(s) != 2 {
			return fmt.Errorf("cannot plot anything other than a 2x1 vector")
		}
		l, err := plotter.NewLine(plotter.XYs{
			{0, 0},
			{s[0], s[1]},
		})
		if err != nil {
			return fmt.Errorf("could not create line :%v", err)
		}
		l.Color = color.RGBA{B: 255, A: 255}

		p.Add(l)
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

//PlotVectorAfterMatrixTransformation plots vector before and after matrix transorfmation
func PlotVectorAfterMatrixTransformation(v matrix.Vector, mat matrix.Matrix, file string) error {
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	if len(v.Slice()) != 2 {
		return fmt.Errorf("cannot plot anything other than a 2x1 vector")
	}
	l, err := plotter.NewLine(plotter.XYs{
		{0, 0},
		{v.Slice()[0], v.Slice()[1]},
	})
	if err != nil {
		return fmt.Errorf("could not create line :%v", err)
	}
	l.Color = color.RGBA{R: 255, A: 255}
	l.Width = 1.0
	vec := v.ApplyMatrix(mat)
	l2, err := plotter.NewLine(plotter.XYs{
		{0, 0},
		{vec.Slice()[0], vec.Slice()[1]},
	})
	if err != nil {
		return fmt.Errorf("could not create line :%v", err)
	}
	l2.Color = color.RGBA{B: 255, A: 255}
	l2.Width = 1.0
	wt, err := p.WriterTo(500, 500, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("could not write to %s :%v", file, err)
	}
	p.Add(l, l2)
	return nil
}

//PlotEigen plots m's eigenvectors
func PlotEigen(m matrix.Matrix, file string) error {
	if m.NumberOfRows() != 2 || m.NumberOfColumns() != 2 {
		return fmt.Errorf("cannot plot anything other than a 2x2 matrix")
	}
	eigens, err := matrix.CalculateEigenvectors2x2(m)
	if err != nil {
		return fmt.Errorf("cannot calculate eigenvectors :%v", err)
	}
	if err := PlotVectors(eigens, file); err != nil {
		return fmt.Errorf("could not plot eigenvectors :%v", err)
	}

	return nil
}

//PlotTangents plots the tangent via gradient to the plot
func PlotTangents(f func(x float64) float64, startingPoint, endPoint float64, iterations int) error {
	var pts Points
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	fi, err := os.Create("Tangents.png")
	if err != nil {
		return fmt.Errorf("could not create file :%v", err)
	}
	iter := (endPoint - startingPoint) / float64(iterations)
	for i := startingPoint; i < endPoint; i += iter {
		grad := GradientAt(f, i)
		po := Point{
			X: i, Y: f(i),
		}
		pts = append(pts, po)
		l, err := plotter.NewLine(plotter.XYs{
			{0, f(0)}, {i, grad*i + f(0)},
		})
		if err != nil {
			return fmt.Errorf("could not create line :%v", err)
		}
		p.Add(l)
	}
	xys := PointToXYs(pts)
	sc, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	sc.Color = color.RGBA{R: 255, A: 255}
	sc.GlyphStyle.Shape = draw.CrossGlyph{}
	p.Add(sc)

	wt, err := p.WriterTo(400, 400, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(fi)
	if err != nil {
		return fmt.Errorf("could not write to writer :%v", err)
	}
	return nil
}

//PlotSinusApproximation plots the sinus approximation onto file
func PlotSinusApproximation(iterations, polynomial int, file string) error {
	if !strings.HasSuffix(file, ".png") {
		return fmt.Errorf("File should be a png file :%s", file)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	defer f.Close()
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	for i := 1; i < polynomial; i++ {
		pts := DefineDatasetPolynomial(SinusEstimate, -1.0, 1.0, iterations, i)
		pts = DiscludeNan(pts)
		xys := PointToXYs(pts)

		s, err := plotter.NewScatter(xys)
		if err != nil {
			return fmt.Errorf("could not create scatter :%v", err)
		}
		s.GlyphStyle.Shape = draw.CrossGlyph{}
		s.Color = color.RGBA{B: 255, A: 255}
		p.Add(s)
	}
	wt, err := p.WriterTo(360, 360, "png")
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
func DrawStddev(pl *plot.Plot, pts Points) error {
	l, err := plotter.NewLine(plotter.XYs{
		{AverageX(pts) - StddevX(pts), AverageY(pts)},
		{AverageX(pts) + StddevY(pts), AverageY(pts)},
	})
	if err != nil {
		return fmt.Errorf("could not draw line :%v", err)
	}
	pl.Add(l)
	return nil
}

//DrawLine draws a line between p1 and p2.
func DrawLine(p1, p2 Point) (*plotter.Line, error) {
	l, err := plotter.NewLine(plotter.XYs{
		{p1.X, p1.Y},
		{p2.X, p2.Y},
	})
	if err != nil {
		return nil, fmt.Errorf("could not draw line :%v", err)
	}
	return l, nil
}

//DrawApproximation draws a line of approximation based on k,n which should be the outputs of ApproximiateLine
func DrawApproximation(pts Points, k, n float64, file string) error {
	max := FindMaxX(pts)
	min := FindMinX(pts)
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create line:  %v", err)
	}
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file:  %v", err)
	}
	if !checkIsNan(min) && !checkIsNan(max) {
		l, err := plotter.NewLine(plotter.XYs{
			{min, min*k + n},
			{max, max*k + n},
		})
		if err != nil {
			return fmt.Errorf("could not create line :%v", err)
		}
		p.Add(l)
	}
	wt, err := p.WriterTo(360, 360, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("could not write to %s :%v", file, err)
	}
	return nil
}

//Integral defines a definite integral between the Points a and b for function f. N represents the number to which you want to divide the space from a and b into. You should be getting better approximations for the area under the curve by increasing n.
func Integral(f func(x float64) float64, a, b float64, n int) float64 {
	area := 0.0
	deltax := (b - a) / float64(n)
	for i := a; i < b; i += deltax {
		area += deltax * f(i)
	}
	return area
}

//IntegralByTrapezoid defines a definite integral between the Points a and b for function f. N represents the number of trapezoids you want to divide the space between a and b into. You should be getting better approximations for the area under the curve by increasing n.
func IntegralByTrapezoid(f func(x float64) float64, a, b float64, n int) float64 {
	area := 0.0
	deltax := (b - a) / float64(n)
	for i := a; i < b; i += deltax {
		base := deltax
		height := (f(i) + f(i+deltax)) / 2
		area += base * height
	}
	return area
}

//MaclaurinFor1Over1MinusX is designed to estimate 1/(1-x) function via Maclaurin series
func MaclaurinFor1Over1MinusX(x float64, polynomial int) float64 {
	approx := 0.0
	for i := 0; i < polynomial; i++ {
		approx += math.Pow(x, float64(i))
	}
	return approx
}

//CosineEstimate returns the Taylor series cosine approximate of X.
func CosineEstimate(x float64, polynomial int) float64 {
	approx := 0.0
	for i := 0; i < polynomial; i++ {
		c := math.Pow(-1, float64(i))
		num := math.Pow(x, float64(2*i))
		denom := Factorial(int(i * 2))
		approx += c * (num / float64(denom))
	}
	return approx
}

//SinusEstimate returns the Taylor series sinus approximate of X.
func SinusEstimate(x float64, polynomial int) float64 {
	approx := 0.0
	var s float64
	var arr []float64
	for i := 0; i < polynomial; i++ {
		if i%2 == 0 {
			s = 0
		} else {
			pow := math.Pow(x, float64(i))
			fact := Factorial(i)
			s = pow / float64(fact)
		}
		arr = append(arr, s)
		for k := range arr {
			if k%2 != 0 {
				s = -s
			}
		}
		approx += s

	}
	return -approx
}

//TanEstimate returns the Taylor series approximation with polynomial accuracy.
func TanEstimate(x float64, polynomial int) float64 {
	return SinusEstimate(x, polynomial) / CosineEstimate(x, polynomial)
}

//ExponentialEstimate returns the Taylor series exponential approximate of X. E to the x.
func ExponentialEstimate(x float64, polynomial int) float64 {
	approx := 0.0
	for i := 0; i < polynomial; i++ {
		comp := math.Pow(x, float64(i))
		fact := Factorial(i)
		approx += comp / float64(fact)
	}
	return approx
}

//HighestAccuracyPolynomial plots the best polynomial approximation of f regarding fa->which should be the actual function. Function returns Points and the polynomial accuracy index.
func HighestAccuracyPolynomial(f func(x float64, polynomial int) float64, fa func(x float64) float64, stPoint, endPoint float64, iterations, uptopolynomial int) ([]Point, int) {
	var ss []float64
	var pts Points
	for i := 1; i < uptopolynomial; i++ {
		pts = DefineDatasetPolynomial(f, stPoint, endPoint, iterations, i)
		s := EstimationError(fa, pts)
		fmt.Println(s)
		ss = append(ss, s)
	}
	x := indexClosest(ss) + 1
	ps := DefineDatasetPolynomial(f, stPoint, endPoint, iterations, x)

	return ps, x
}

//GradientLines plots the tangents based on the gradients of the function f, given the starting and end point and the step
func GradientLines(f func(float64) float64, stPoint, endPoint float64, step float64, file string) error {
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot :%v", err)
	}
	fi, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s :%v", file, err)
	}
	defer fi.Close()
	iterations := int((endPoint - stPoint) / step)
	points := DefineDataset(f, stPoint, endPoint, iterations)
	xys := PointToXYs(points)
	sc, err := plotter.NewScatter(xys)
	if err != nil {
		return fmt.Errorf("could not create scatter :%v", err)
	}
	sc.GlyphStyle.Shape = draw.CrossGlyph{}
	sc.Color = color.RGBA{B: 255, A: 255}
	p.Add(sc)
	for i := stPoint; i < endPoint; i += step {
		grad := (f(i+step) - f(i)) / step
		n := (endPoint - stPoint) / 4
		l, err := plotter.NewLine(plotter.XYs{
			{i - n, f(i) - grad*n}, {i + step + n, f(i+step) + grad*n},
		})
		if err != nil {
			return fmt.Errorf("could not create line :%v", err)
		}
		p.Add(l)
	}
	wt, err := p.WriterTo(400, 400, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(fi)
	if err != nil {
		return fmt.Errorf("could not write to writer :%v", err)
	}
	return nil
}
