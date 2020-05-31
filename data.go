//Package plot implements plotting functionality for all sorts of data as well as matrix and vector visualisations. It also provides statistical components of datasets, such as variances and covariances.
package plot

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/timothy102/matrix"
	"gonum.org/v1/plot/plotter"
)

type point struct {
	x, y float64
}

//PrintPoints prints points
func PrintPoints(points []point) {
	for i, p := range points {
		fmt.Printf("%d : (%.2f, %.2f)\n", i+1, p.x, p.y)
	}
}

//RoundPoints rounds every floating number in points
func RoundPoints(points []point) []point {
	for _, p := range points {
		p.x = math.Round(p.x)
		p.y = math.Round(p.y)
	}
	return points
}

//DefineDataset returns an array of points given the inputs. The function will iterate from stPoint to endPoint with iterations.
func DefineDataset(f func(x float64) float64, stPoint, endPoint float64, iterations int) []point {
	var points []point
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := point{
			x: i,
			y: f(i),
		}
		points = append(points, p)
	}
	return points
}

//RandomPoints returns 'number' of points between min and max
func RandomPoints(number int, min, max float64) []point {
	var points []point
	for i := 0; i < number; i++ {
		p := point{
			x: min + rand.Float64()*(max-min),
			y: min + rand.Float64()*(max-min),
		}
		points = append(points, p)
	}

	return points
}
func randFloats(min, max float64, n int) []float64 {
	fls := make([]float64, n)
	for i := range fls {
		fls[i] = min + rand.Float64()*(max-min)
	}
	return fls
}

//DatasetWithRandomNoise generates a dataset based on a function and random noise
func DatasetWithRandomNoise(f func(x float64) float64, stPoint, endPoint float64, iterations int) []point {
	rand.Seed(time.Now().UnixNano())
	var points []point
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		noise := float64(rand.Intn(iterations/2)) * 0.2
		p := point{
			x: i,
			y: f(i/noise) * noise,
		}
		points = append(points, p)
	}
	return points
}

//DatasetWithSetNoise generates a dataset based on a function and input noise
func DatasetWithSetNoise(f func(x float64) float64, noise float64, stPoint, endPoint float64, iterations int) []point {
	rand.Seed(time.Now().UnixNano())
	var points []point
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := point{
			x: i,
			y: f(i) + noise,
		}
		points = append(points, p)
	}
	return points
}

//DefineDatasetQuadratic defines a quadratic dataset
func DefineDatasetQuadratic(f func(x, a, b, c float64) float64, a, b, c float64, stPoint, endPoint float64, iterations int) []point {
	var points []point
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := point{
			x: i,
			y: f(i, a, b, c),
		}
		points = append(points, p)
	}
	return points
}

//Season struct is used with the MessyDataset function to generate a seasonal plot. Start and end are defining an interval where your seasonality would like to occur.
type Season struct {
	bias, time, slope, amplitude, noiseLevel float64
	stPoint, endPoint                        float64
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
func Series(i Season, x float64) float64 {
	ser := i.bias + trend(i.time, i.slope) + Seasonality(i.time, i.amplitude, 10, 1)
	ser += i.noiseLevel
	return ser
}

func SeriesDataset(s Season, iterations int) []point {
	var points []point
	iter := (s.endPoint - s.stPoint) / float64(iterations)
	for i := s.stPoint; i <= s.endPoint; i += iter {
		p := point{
			x: i,
			y: Series(s, i),
		}
		points = append(points, p)
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
	fmt.Printf("Std. deviation of X : %.2f\n", stddevX)
	fmt.Printf("Std. deviation of Y : %.2f\n", stddevY)
	fmt.Printf("Covariance : %.5f\n", cov)

	CovarianceMatrix(points)
	Correlation(cov)
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

//ApproximateLine approxiamates the line via gradient descent
func ApproximateLine(points []point, learningRate float64, iterations int) (k, n float64) {
	for i := 0; i < 10000; i++ {
		cost, dm, dc := Gradient(points, k, n)
		k += -dm * learningRate
		n += -dc * learningRate
		if (10 * i % iterations) == 0 {
			fmt.Printf("cost(%.2f,%.2f)=%.2f\n", k, n, cost)
		}
	}
	return k, n
}

//Gradient returns the vector gradients of points
func Gradient(points []point, k, n float64) (float64, float64, float64) {
	var cost, dm, dc float64
	for i := range points {
		dist := points[i].y - (points[i].x*k + n)
		dm += points[i].x * dist
		dc += -dist
	}
	l := float64(len(points))
	return cost / l, 2 / l * dm, 2 / n * dc
}

//FindMax returns the point with the highest X value
func FindMaxX(points []point) float64 {
	max := 0.5
	for _, p := range points {
		if p.x > max {
			max = p.x
		}
	}
	return max
}

//FindMaxY returns the point with the highest Y value
func FindMaxY(points []point) float64 {
	max := 0.5
	for _, p := range points {
		if p.y > max {
			max = p.y
		}
	}
	return max
}

//FindMinX returns the point with the lowest X value
func FindMinX(points []point) float64 {
	min := 0.5
	for _, p := range points {
		if p.x < min {
			min = p.x
		}
	}
	return min
}

//FindMinY returns the point with the lowest Y value
func FindMinY(points []point) float64 {
	min := 0.5
	for _, p := range points {
		if p.y < min {
			min = p.y
		}
	}
	return min
}

//AbsoluteError returns the error
func AbsoluteError(f func(x float64) float64, points []point) float64 {
	var loss float64
	var e float64
	for _, p := range points {
		e = p.y - f(p.x)
		loss += e
	}
	return loss
}

//MSE returns the mean squared error
func MSE(f func(x float64) float64, points []point) float64 {
	var loss float64
	var e float64
	for _, p := range points {
		e = math.Pow(p.y-f(p.x), 2)
		loss += e
	}
	return loss
}

//RMSE returns the root mean squared error
func RMSE(f func(x float64) float64, points []point) float64 {
	return math.Sqrt(MSE(f, points))

}

//CrossEntropy returns the cross entropy loss
func CrossEntropy(f func(x float64) float64, points []point) float64 {
	var loss float64
	for _, p := range points {
		loss += p.y*math.Log(f(p.x)) + (1-p.y)*math.Log(1-p.x)
	}
	return loss
}
