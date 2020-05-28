package plot

import (
	"bufio"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/timothy102/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

type point struct {
	x, y float64
}

//DefineDataset returns an array of points given the inputs. The function will iterate from stPoint to endPoint with a step of 1/iterations.
func DefineDataset(f func(x float64) float64, iterations int, stPoint, endPoint float64) []point {
	var points []point
	for i := stPoint; i < endPoint; i += 1 / float64(iterations) {
		p := point{
			x: i,
			y: f(i),
		}
		points = append(points, p)
	}
	return points
}
func RandomPoints(number int, downlimit, uplimit float64) []point {
	var points []point
	for i := 0; i < number; i++ {
		rand.Seed(time.Now().UnixNano())
		k := rand.Float64()
		n := rand.Float64()

		if k > downlimit || n > downlimit || k < uplimit || n < uplimit {
			continue
		} else {
			p := point{
				x: float64(k) * 0.2,
				y: float64(n) * 0.2,
			}
			points = append(points, p)
		}
	}
	return points
}

//Seasonality struct is used with the MessyDataset function to generate a seasonal plot. Start and end are defining an interval where your seasonality would like to occur.
type Interval struct {
	start, end float64
	frequency  float64
	noise      float64
}

func trend(time, slope float64) float64 {
	return slope * time
}

//SeasonalPattern defines what the function is on any interval
func SeasonalPattern(time float64) float64 {
	if time < 0.1 {
		return math.Cos(time * 7 * math.Pi)
	} else {
		return 1 / math.Exp(5*time)
	}
}

//Seasonality returns the amplitude* seasonalPattern
func Seasonality(time, amplitude float64, period, phase int) float64 {
	seasonTime := ((int(time) + phase) % period) / period
	return amplitude * SeasonalPattern(float64(seasonTime))
}

//Series returns a seasonal plot defined by 5 parameters.
//X is the actual input, time and slope define the gradient, amplitude is the frequency and noiseLevel is a constant
//You can use the SeasonalPattern,Seasonality and trend to define this function, but this has been taken care of for you.
func Series(x, bias, time, slope, amplitude, noiseLevel float64) float64 {
	time = 100.0
	ser := bias + trend(time, slope) + Seasonality(time, amplitude, 100, 1)
	ser += noiseLevel
	return ser
}

//Messy dataset generates a messy dataset, based on a function. It iterates from stPoint to endPoint along with the seasonality struct.
func MessyDataset(f func(x float64) float64, in *Interval, stPoint, endPoint float64) []point {
	var points []point
	for i := stPoint; i < endPoint; i += 0.001 {
		if i < in.end && i > in.start {
			p := point{
				x: i,
				y: f(i)*in.frequency + in.noise,
			}
			points = append(points, p)

		} else {
			p := point{
				x: i,
				y: f(i),
			}
			points = append(points, p)
		}
	}
	return points
}

//ComputeAverageX computes the average of x coordinates of the dataset
func ComputeAverageX(points []point) float64 {
	var sum float64
	for _, p := range points {
		sum += p.x
	}
	avg := sum / float64(len(points))
	return avg
}

//ComputeAverageY computes the average of y coordinates of the dataset
func ComputeAverageY(points []point) float64 {
	var sum float64
	for _, p := range points {
		sum += p.y
	}
	avg := sum / float64(len(points))
	return avg
}

//Gaussian returns the Gaussian distribution function. For plotting, check PlotGaussian().
func Gaussian(x float64, mean float64, stddev float64) float64 {
	s := math.Pow((x-mean)/stddev, 2)
	return 1 / stddev * math.Sqrt(2*math.Pi) * math.Exp(-0.5*s)
}

//Sigmoid returns the sigmoid of x
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//ShiftDatasetOnX shifts the x coordinates by the scalar
func ShiftDatasetOnX(points []point, scalar float64) {
	for _, p := range points {
		p.x += scalar
	}
}

//ShiftDatasetOnY shifts the y coordinates by the scalar
func ShiftDatasetOnY(points []point, scalar float64) []point {
	for _, p := range points {
		p.y += scalar
	}
	return points
}

//StretchByFactor streches the y coordinates by the factor. Check how mean and variance change.
func StretchByFactor(points []point, factor float64) []point {
	for _, p := range points {
		p.y *= factor
	}
	return points
}

//ComputeVarianceX returns the variance of X coordinates
func ComputeVarianceX(points []point) float64 {
	var sum float64
	avg := ComputeAverageX(points)
	for _, p := range points {
		sum += math.Pow(p.x-avg, 2)
	}
	return sum / float64(len(points))
}

//ComputeVarianceY returns the variance of Y coordinates
func ComputeVarianceY(points []point) float64 {
	var sum float64
	avg := ComputeAverageY(points)
	for _, p := range points {
		sum += math.Pow(p.y-avg, 2)
	}
	return sum / float64(len(points))
}

//ComputeCovariance returns the covariance of a given dataset
func ComputeCovariance(points []point) float64 {
	var cov float64
	avgX := ComputeAverageX(points)
	avgY := ComputeAverageY(points)

	for _, p := range points {
		cov += (p.x - avgX) * (p.y - avgY)
	}
	return cov / float64(len(points))
}

//ComputeStddevX returns the standard devation of X coordinates
func ComputeStddevX(points []point) float64 {
	return math.Sqrt(ComputeVarianceX(points))
}

//ComputeStddevY returns the standard devation of Y coordinates
func ComputeStddevY(points []point) float64 {
	return math.Sqrt(ComputeVarianceY(points))
}

//Correlation prints the memo for how the X and Y of the dataset are correlated. Input should be the covariance.
func Correlation(cov float64) {
	if cov > 0 {
		fmt.Println("X and Y are positively correlated. ")
	} else if cov == 0 {
		fmt.Println("X and Y are not correlated. ")
	} else {
		fmt.Println("X and Y are negatively correlated. ")
	}
}

//CovarianceMatrix returns the covariance matrix of the dataset.
func CovarianceMatrix(points []point) {
	varX := ComputeVarianceX(points)
	varY := ComputeVarianceY(points)
	cov := ComputeCovariance(points)

	slc := [][]float64{
		{varX, cov},
		{cov, varY},
	}
	covMatrix, err := matrix.NewMatrix(slc, 2, 2)
	if err != nil {
		log.Fatalf("could not create covariance matrix :%v", err)
	}
	covMatrix.PrintByRow()
}

//ReadFromDatafile reads from the filepath and returns an array of points.
func ReadFromDatafile(filepath string) ([]point, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("could not open %s:%v", filepath, err)
	}
	defer f.Close()
	var points []point
	s := bufio.NewScanner(f)
	for s.Scan() {
		var x, y float64
		_, err := fmt.Sscanf(s.Text(), "%f,%f", &x, &y)
		if err != nil {
			log.Printf("discarding bad data point %q: %v", s.Text(), err)
			continue
		}
		p := point{
			x: x,
			y: y,
		}
		points = append(points, p)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("could not scan :%v", err)
	}
	return points, nil

}

//DisplayEverything prints all the statistics considering the dataset.
func DisplayEverything(points []point) {
	avgX := ComputeAverageX(points)
	avgY := ComputeAverageY(points)
	varX := ComputeVarianceX(points)
	varY := ComputeVarianceY(points)
	stddevX := ComputeStddevX(points)
	stddevY := ComputeStddevY(points)
	cov := ComputeCovariance(points)

	fmt.Printf("Average of X : %.2f\n", avgX)
	fmt.Printf("Average of Y : %.2f\n", avgY)
	fmt.Printf("Variance of X : %.2f\n", varX)
	fmt.Printf("Variance of Y : %.2f\n", varY)
	fmt.Printf("Std.deviation of X : %.2f\n", stddevX)
	fmt.Printf("Std.deviation of Y : %.2f\n", stddevY)
	fmt.Printf("Covariance : %.5f\n", cov)

	CovarianceMatrix(points)
	Correlation(cov)
}

//PlotMatrix returns the input matrix plotted to a file.
func PlotMatrix(m matrix.Matrix, file string) error {
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("could not create file %s:%v", file, err)
	}
	defer f.Close()
	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot")
	}
	wt, err := p.WriterTo(500, 500, "png")
	if err != nil {
		return fmt.Errorf("could not create writer :%v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("could not write to %s :%v", file, err)
	}
	if m.NumberOfRows() > 2 || m.NumberOfColumns() > 2 {
		return fmt.Errorf("Matrix you're trying to plot should be a 2x2 matrix.")
	}
	var points []point
	ps := RandomPoints(10, -1.0, 1.0)

	vecs := PointToVector(ps)
	for _, v := range vecs {
		if v.GetLength() == 1 {
			p := point{
				x: v.Slice()[0],
				y: v.Slice()[1],
			}
			points = append(points, p)
		}

	}
	xys := PointToXYs(points)

	//scatter for the original matrix vectors,
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

//PointoVector is a helper function
func PointToVector(points []point) []matrix.Vector {
	var vectors []matrix.Vector
	var vec matrix.Vector
	for i, p := range points {
		vec = vectors[i]
		vec.Slice()[0] = p.x
		vec.Slice()[1] = p.y
	}
	vectors = append(vectors, vec)
	return vectors
}

//PointToXYs is a helper function
func PointToXYs(points []point) plotter.XYs {
	xys := make(plotter.XYs, len(points))
	for i, p := range points {
		xys[i].X = p.x
		xys[i].Y = p.y

	}
	return xys
}

//VectorToPoint is a helper function
func VectorToPoint(vectors []matrix.Vector) []point {
	points := make([]point, len(vectors))
	for _, v := range vectors {
		p := point{
			x: v.Slice()[0],
			y: v.Slice()[1],
		}
		points = append(points, p)
	}
	return points
}

//VectorToXYs is a helper function
func VectorToXYs(vectors []matrix.Vector) plotter.XYs {
	xys := make(plotter.XYs, len(vectors))
	for i, v := range vectors {
		xys[i].X = v.Slice()[0]
		xys[i].Y = v.Slice()[1]
	}
	return xys
}
