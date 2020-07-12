package plot

import (
	"fmt"
	"math"
)

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

//EulerNumberEstimate returns the Taylor series exponential approximate of X.
func EulerNumberEstimate(x float64, polynomial int) float64 {
	approx := 0.0
	for i := 0; i < polynomial; i++ {
		comp := math.Pow(x, float64(i))
		fact := Factorial(i)
		approx += comp / float64(fact)
	}
	return approx
}

//TanEstimate returns the Taylor series approximation with polynomial accuracy.
func TanEstimate(x float64, polynomial int) float64 {
	return SinusEstimate(x, polynomial) / CosineEstimate(x, polynomial)
}

//HighestAccuracyPolynomial plots the best polynomial approximation of f regarding fa->which should be the actual function. Function returns Points and the polynomial accuracy index.
func HighestAccuracyPolynomial(f func(x float64, polynomial int) float64, fa func(x float64) float64, stPoint, endPoint float64, iterations, uptopolynomial int) ([]Point, int) {
	var ss []float64
	var pts Points
	for i := 1; i < uptopolynomial; i++ {
		pts = DefineDatasetWithPolynomial(f, stPoint, endPoint, iterations, i)
		s := EstimationError(fa, pts)
		fmt.Println(s)
		ss = append(ss, s)
	}
	x := indexClosest(ss) + 1
	ps := DefineDatasetWithPolynomial(f, stPoint, endPoint, iterations, x)

	return ps, x
}
