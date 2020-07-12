package plot

import (
	"fmt"
	"math"

	"github.com/timothy102/matrix"
)

//AverageX computes the average of x coordinates of the dataset
func AverageX(pts Points) float64 {
	var sum float64
	for _, p := range pts {
		sum += p.x
	}
	avg := sum / float64(len(pts))
	return avg
}

//AverageY computes the average of y coordinates of the dataset
func AverageY(pts Points) float64 {
	var sum float64
	for _, p := range pts {
		sum += p.y
	}
	avg := sum / float64(len(pts))
	return avg
}

//VarianceX returns the variance of X coordinates
func VarianceX(pts Points) float64 {
	var sum float64
	avg := AverageX(pts)
	for _, p := range pts {
		sum += math.Pow(p.x-avg, 2)
	}
	return sum / float64(len(pts))
}

//VarianceY returns the variance of Y coordinates
func VarianceY(pts Points) float64 {
	var sum float64
	avg := AverageY(pts)
	for _, p := range pts {
		sum += math.Pow(p.y-avg, 2)
	}
	return sum / float64(len(pts))
}

//Covariance returns the covariance of a given dataset
func Covariance(pts Points) float64 {
	var cov float64
	avgX := AverageX(pts)
	avgY := AverageY(pts)

	for _, p := range pts {
		cov += (p.x - avgX) * (p.y - avgY)
	}
	return cov / float64(len(pts))
}

//StddevX returns the standard devation of X coordinates
func StddevX(pts Points) float64 {
	return math.Sqrt(VarianceX(pts))
}

//StddevY returns the standard devation of Y coordinates
func StddevY(pts Points) float64 {
	return math.Sqrt(VarianceY(pts))
}

//Correlation prints the memo for how the X and Y of the dataset are correlated.
func Correlation(pts Points) {
	cov := Covariance(pts)
	if cov > 0 {
		fmt.Println("X and Y are positively correlated. ")
	} else if cov == 0 {
		fmt.Println("X and Y are not correlated. ")
	} else {
		fmt.Println("X and Y are negatively correlated. ")
	}
}

//CovarianceMatrix returns the covariance matrix of the dataset.
func CovarianceMatrix(pts Points) matrix.Matrix {
	varX := VarianceX(pts)
	varY := VarianceY(pts)
	cov := Covariance(pts)

	slc := [][]float64{
		{varX, cov},
		{cov, varY},
	}
	covMatrix := matrix.NewMatrix(slc)
	return covMatrix
}

//DisplayAllStats prints all the statistics considering the dataset.
func DisplayAllStats(pts Points) {
	avgX := AverageX(pts)
	avgY := AverageY(pts)
	varX := VarianceX(pts)
	varY := VarianceY(pts)
	stddevX := StddevX(pts)
	stddevY := StddevY(pts)
	cov := Covariance(pts)

	fmt.Printf("Average of X : %.2f\n", avgX)
	fmt.Printf("Average of Y : %.2f\n", avgY)
	fmt.Printf("Variance of X : %.2f\n", varX)
	fmt.Printf("Variance of Y : %.2f\n", varY)
	fmt.Printf("Standard deviation of X : %.2f\n", stddevX)
	fmt.Printf("Standard deviation of Y : %.2f\n", stddevY)
	fmt.Printf("Covariance : %.5f\n", cov)

	CovarianceMatrix(pts)
	Correlation(pts)
}
