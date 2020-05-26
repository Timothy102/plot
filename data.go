package plot

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/timothy102/matrix"
)

type point struct {
	x, y float64
}

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
func ComputeAverageX(points []point) float64 {
	var sum float64
	for _, p := range points {
		sum += p.x
	}
	avg := sum / float64(len(points))
	return avg
}
func ComputeAverageY(points []point) float64 {
	var sum float64
	for _, p := range points {
		sum += p.y
	}
	avg := sum / float64(len(points))
	return avg
}
func Gaussian(x float64, mean float64, stddev float64) float64 {
	s := math.Pow((x-mean)/stddev, 2)
	return 1 / stddev * math.Sqrt(2*math.Pi) * math.Exp(-0.5*s)
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func ShiftDatasetOnX(points []point, scalar float64) {
	for _, p := range points {
		p.x += scalar
	}
}
func ShiftDatasetOnY(points []point, scalar float64) []point {
	for _, p := range points {
		p.y += scalar
	}
	return points
}
func StretchByFactor(points []point, factor float64) []point {
	for _, p := range points {
		p.y *= factor
	}
	return points
}
func ComputeVarianceX(points []point) float64 {
	var sum float64
	avg := ComputeAverageX(points)
	for _, p := range points {
		sum += math.Pow(p.x-avg, 2)
	}
	return sum / float64(len(points))
}
func ComputeVarianceY(points []point) float64 {
	var sum float64
	avg := ComputeAverageY(points)
	for _, p := range points {
		sum += math.Pow(p.y-avg, 2)
	}
	return sum / float64(len(points))
}
func ComputeCovariance(points []point) float64 {
	var cov float64
	avgX := ComputeAverageX(points)
	avgY := ComputeAverageY(points)

	for _, p := range points {
		cov += (p.x - avgX) * (p.y - avgY)
	}
	return cov / float64(len(points))
}
func ComputeStddevX(points []point) float64 {
	return math.Sqrt(ComputeVarianceX(points))
}
func ComputeStddevY(points []point) float64 {
	return math.Sqrt(ComputeVarianceY(points))
}
func Correlation(cov float64) {
	if cov > 0 {
		fmt.Println("X and Y are positively correlated. ")
	} else if cov == 0 {
		fmt.Println("X and Y are not correlated. ")
	} else {
		fmt.Println("X and Y are negatively correlated. ")
	}
}
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
func DisplayEverything(points []point) {
	avgX := ComputeAverageX(points)
	avgY := ComputeAverageY(points)
	varX := ComputeVarianceX(points)
	varY := ComputeVarianceY(points)
	stddevX := ComputeStddevX(points)
	stddevY := ComputeStddevY(points)
	cov := ComputeCovariance(points)

	fmt.Println("Average of X :", avgX)
	fmt.Println("Average of Y :", avgY)
	fmt.Println("Variance of X :", varX)
	fmt.Println("Variance of Y :", varY)
	fmt.Println("Std.deviation of X :", stddevX)
	fmt.Println("Std.deviation of Y :", stddevY)
	fmt.Println("Covariance :", cov)

	CovarianceMatrix(points)
	Correlation(cov)
}
