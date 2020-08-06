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

//Point struct
type Point struct {
	x, y   float64
	isBlue bool
}

//Points is a slice of Points.
type Points []Point

//PrintPoints prints Points by line.
func PrintPoints(pts Points) {
	for i, p := range pts {
		fmt.Printf("%d : (%.2f, %.2f)\n", i+1, p.x, p.y)
	}
}

//NewPoint returns a new point object.
func NewPoint(x, y float64) Point {
	return Point{x: x, y: y}
}

//RoundPointstoDecimals rounds all Points to decimals accuracy.
func RoundPointstoDecimals(pts Points, decimals int) Points {
	for _, p := range pts {
		p.x = roundTo(p.x, decimals)
		p.y = roundTo(p.y, decimals)
	}
	return pts
}

//Randomize randomizes points to random values.
func Randomize(pts Points) Points {
	fls := randFloats(-5.0, 5.0, 2*len(pts))
	for i := range pts {
		pts[i].x = fls[i]
		pts[i].y = fls[len(fls)-i]
	}
	return pts
}

//Normalize normalizes points so that they are centered and the standard deviation is 1.
func Normalize(pts Points) Points {
	for i := range pts {
		pts[i].x = (pts[i].x - AverageX(pts)) / StddevX(pts)
		pts[i].y = (pts[i].y - AverageY(pts)) / StddevY(pts)
	}
	return pts
}
func roundTo(number float64, decimals int) float64 {
	s := math.Pow(10, float64(decimals))
	return math.Round(number*s) / s
}
func swap(pts Points, index int) {
	val := pts[index]
	pts[index] = pts[index-1]
	pts[index-1] = val
}

func takeAverage(fs []float64) float64 {
	sum := 0.0
	for _, f := range fs {
		sum += f
	}
	return sum / float64(len(fs))
}

func xToArray(pts Points) []float64 {
	arr := make([]float64, len(pts))
	for _, p := range pts {
		arr = append(arr, p.x)
	}
	return arr
}

func yToArray(pts Points) []float64 {
	arr := make([]float64, len(pts))
	for _, p := range pts {
		arr = append(arr, p.y)
	}
	return arr
}

func max(fs []float64) float64 {
	max := 1.0
	for _, f := range fs {
		if f > max {
			max = f
		}
	}
	return max
}
func min(fs []float64) float64 {
	min := 0.5
	for _, f := range fs {
		if f < min {
			min = f
		}
	}
	return min
}
func indexOfMin(fs []float64) int {
	m := min(fs)
	for i, f := range fs {
		if m == f {
			return i
		}
	}
	return 0
}

func indexOfMax(fs []float64) int {
	m := max(fs)
	for i, f := range fs {
		if m == f {
			return i
		}
	}
	return 0
}
func closestzero(fls []float64) float64 {
	m := 10.0
	for _, f := range fls {
		abs := Absolute(f)
		if abs < m {
			m = abs
		}
	}
	return m
}

func indexClosest(fls []float64) int {
	zero := closestzero(fls)
	for i, f := range fls {
		if f == zero {
			return i
		}
	}
	return 0
}
func indexOfY(pts Points, y float64) int {
	for i, p := range pts {
		if p.y == y {
			return i
		}
	}
	return 0
}
func indexOfX(pts Points, x float64) int {
	for i, p := range pts {
		if p.x == x {
			return i
		}
	}
	return 0
}

//ClosestToZero returns value closest to zero. You can use this function to  see which approximation best estimates the actual function.
func ClosestToZero(pts Points) float64 {
	zero := 10.0
	for _, f := range pts {
		abs := Absolute(f.y)
		if abs < zero {
			zero = abs
		}
	}
	return zero
}

//IndexAtClosestToZero returns the index of  the value within Points that is closest to zero.
func IndexAtClosestToZero(pts Points) int {
	zero := ClosestToZero(pts)
	return indexOfY(pts, zero)
}

//PointNearestValue returns the Point which is closest to value on the Y axis
func PointNearestValue(pts Points, value float64) Point {
	nearest := 100.0
	for i, p := range pts {
		v := p.y - value
		v = Absolute(v)
		if v < nearest {
			nearest = v
		}
		return GetPoint(pts, i)
	}
	return Point{}
}

//IndexAtClosestToValue returns the index of the Point that is closest to the value on the Y axis.
func IndexAtClosestToValue(pts Points, value float64) int {
	p := PointNearestValue(pts, value)
	return GetIndex(pts, p)
}

//Euclidean returns the Euclidean distance between two Points.
func Euclidean(p1, p2 Point) float64 {
	s := math.Pow((p1.x - p2.x), 2)
	k := math.Pow((p1.y - p2.y), 2)
	return math.Sqrt(s + k)
}

//AverageEuclidean returns the average euclidean distance by iteration through all Points.
func AverageEuclidean(pts Points) float64 {
	sum := 0.0
	for i := range pts {
		if i != len(pts)-1 {
			sum += Euclidean(pts[i], pts[i+1])
		}
	}
	return sum / float64(len(pts))
}

//FlipOverXAxis flips Points over the X axis.
func FlipOverXAxis(pts Points) Points {
	for i := range pts {
		pts[i].x = -pts[i].x
	}
	return pts
}

//FlipOverYAxis flips Points over the Y axis.
func FlipOverYAxis(pts Points) Points {
	for i := range pts {
		pts[i].y = -pts[i].y
	}
	return pts
}

//SortPointsByX bubble sorts Points, so the Point with the highest X value is the last element of the array.
func SortPointsByX(pts Points) Points {
	for i := len(pts); i > 0; i-- {
		for j := 1; j < i; j++ {
			if pts[j-1].x > pts[j].x {
				swap(pts, j)
			}
		}
	}
	return pts
}

//SortPointsByY bubble sorts Points, so the Point with the highest Y value is the last element of the array.
func SortPointsByY(pts Points) Points {
	for i := len(pts); i > 0; i-- {
		for j := 1; j < i; j++ {
			if pts[j-1].y > pts[j].y {
				swap(pts, j)
			}
		}
	}
	return pts
}

//DefineDataset returns  Points given the inputs. The function will iterate from stPoint to endPoint producing iterations number of Points.
func DefineDataset(f func(x float64) float64, stPoint, endPoint float64, iterations int) Points {
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := Point{
			x: i,
			y: f(i),
		}
		pts = append(pts, p)
	}
	return pts
}

//DefineDatasetWithPolynomial returns an array of Points given the inputs. The function will iterate from stPoint to endPoint with iterations.
func DefineDatasetWithPolynomial(f func(x float64, polynomial int) float64, stPoint, endPoint float64, iterations, polynomial int) Points {
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := Point{
			x: i,
			y: f(i, polynomial),
		}
		pts = append(pts, p)
	}
	return pts
}

//DefineDatasetToPower  is designed to plot Points to the power datasets.
func DefineDatasetToPower(f func(x float64, n float64) float64, n float64, stPoint, endPoint float64, iterations int) Points {
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := Point{
			x: i,
			y: f(i, n),
		}
		pts = append(pts, p)
	}
	return pts
}

//DefineRandomPoints returns 'number' of Points between min and max
func DefineRandomPoints(number int, min, max float64) Points {
	var pts Points
	for i := 0; i < number; i++ {
		p := Point{
			x: min + rand.Float64()*(max-min),
			y: min + rand.Float64()*(max-min),
		}
		pts = append(pts, p)
	}

	return pts
}

//DefineWithRandomNoise generates a dataset based on a function and random noise
func DefineWithRandomNoise(f func(x float64) float64, stPoint, endPoint float64, iterations int) Points {
	rand.Seed(time.Now().UnixNano())
	var noise float64
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		noise = rand.Float64()
		p := Point{
			x: i,
			y: f(i) * noise,
		}

		//fmt.Printf("%.3f\n", noise)
		pts = append(pts, p)
	}
	return pts
}

//DefineDatasetQuadratic defines a quadratic dataset
func DefineDatasetQuadratic(f func(x, a, b, c float64) float64, a, b, c float64, stPoint, endPoint float64, iterations int) Points {
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := Point{
			x: i,
			y: f(i, a, b, c),
		}
		pts = append(pts, p)
	}
	return pts
}

func randFloats(min, max float64, n int) []float64 {
	fls := make([]float64, n)
	for i := range fls {
		fls[i] = min + rand.Float64()*(max-min)
	}
	return fls
}

//Season struct is used with the MessyDataset function to generate a seasonal plot. Start and end are defining an interval where your seasonality would like to occur.
type Season struct {
	bias, time, slope, amplitude, noiseLevel float64
	stPoint, endPoint                        float64
}

func trend(time, slope float64) float64 {
	return slope * time
}

func seasonalPattern(time float64) float64 {
	if time < 0.1 {
		return math.Cos(time * 7 * math.Pi)
	}
	return 1 / math.Exp(5*time)

}

//Seasonality returns the amplitude* seasonalPattern
func Seasonality(time, amplitude float64, period, phase int) float64 {
	seasonTime := ((int(time) + phase) % period) / period
	return amplitude * seasonalPattern(float64(seasonTime))
}

//Series returns a seasonal plot defined by 5 parameters.
//X is the actual input, time and slope define the gradient, amplitude is the frequency and noiseLevel is a constant
//You can use the SeasonalPattern,Seasonality and trend to define this function, but this has been taken care of for you.
func Series(i Season, x float64) float64 {
	ser := i.bias + trend(i.time, i.slope) + Seasonality(i.time, i.amplitude, 10, 1)
	ser += i.noiseLevel
	return ser
}

//SeriesDataset returns Points from inputs and iterations.
func SeriesDataset(s Season, iterations int) Points {
	var pts Points
	iter := (s.endPoint - s.stPoint) / float64(iterations)
	for i := s.stPoint; i <= s.endPoint; i += iter {
		p := Point{
			x: i,
			y: Series(s, i),
		}
		pts = append(pts, p)
	}
	return pts
}

//Sigmoid returns the sigmoid of x
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//SigmoidPrime returns the  derivative of sigmoid.
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

//Absolute returns absolute value of x
func Absolute(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

//Gaussian returns the Gaussian distribution function. For plotting, check PlotGaussian().
func Gaussian(x float64, mean float64, stddev float64) float64 {
	s := math.Pow((x-mean)/stddev, 2)
	return 1 / stddev * math.Sqrt(2*math.Pi) * math.Exp(-0.5*s)
}

//ShiftDatasetOnX shifts the x coordinates by the scalar
func ShiftDatasetOnX(pts Points, scalar float64) Points {
	for i := range pts {
		pts[i].x = pts[i].x + scalar
	}
	return pts
}

//ShiftDatasetOnY shifts the y coordinates by the scalar
func ShiftDatasetOnY(pts Points, scalar float64) Points {
	for i := range pts {
		pts[i].y = pts[i].y + scalar
	}
	return pts
}

//StretchByFactorX streches the x coordinates by the factor. Check how mean and variance change.
func StretchByFactorX(pts Points, factor float64) Points {
	for i := range pts {
		pts[i].x = pts[i].x * factor
	}
	return pts
}

//StretchByFactorY streches the x coordinates by the factor. Check how mean and variance change.
func StretchByFactorY(pts Points, factor float64) Points {
	for i := range pts {
		pts[i].y = pts[i].y * factor
	}
	return pts
}

//ReadFromDatafile reads from the filepath and returns an array of Points.
func ReadFromDatafile(filepath string) (Points, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("could not open %s:%v", filepath, err)
	}
	defer f.Close()
	var pts Points
	s := bufio.NewScanner(f)
	for s.Scan() {
		var x, y float64
		_, err := fmt.Sscanf(s.Text(), "%f,%f", &x, &y)
		if err != nil {
			log.Printf("discarding bad data Point %q: %v", s.Text(), err)
			continue
		}
		p := Point{
			x: x,
			y: y,
		}
		pts = append(pts, p)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("could not scan :%v", err)
	}
	return pts, nil

}

//PointToVector is a helper function
func PointToVector(pts Points) []matrix.Vector {
	var vectors []matrix.Vector
	for i := range pts {
		slc := []float64{0.0, 0.0}
		vec := matrix.NewVector(slc)
		vec.Slice()[0] = pts[i].x
		vec.Slice()[1] = pts[i].y
		vectors = append(vectors, vec)
	}
	return vectors
}

//PointToXYs is a helper function
func PointToXYs(pts Points) plotter.XYs {
	xys := make(plotter.XYs, len(pts))
	for i, p := range pts {
		xys[i].X = p.x
		xys[i].Y = p.y
	}

	return xys
}

//VectorToPoints is a helper function
func VectorToPoints(vectors []matrix.Vector) Points {
	pts := make(Points, len(vectors))
	for _, v := range vectors {
		p := Point{
			x: v.Slice()[0],
			y: v.Slice()[1],
		}
		pts = append(pts, p)
	}
	return pts
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

func checkIsNan(x float64) bool {
	return math.IsNaN(x)
}

//DiscludeNan gets rid of every NaN dataPoint
func DiscludeNan(pts Points) Points {
	ps := make(Points, len(pts))
	for _, p := range pts {
		if !checkIsNan(p.x) && !checkIsNan(p.y) {
			ps = append(ps, p)
		} else {
			log.Println("Discarding bad dataPoint. ")
		}
	}
	return ps
}

//GetIndex returns the index of p in Points
func GetIndex(pts Points, p Point) (x int) {
	if !PointInDataset(pts, p) {
		log.Printf("Point not in dataset")
	}
	for i := range pts {
		if pts[i].x == p.x && pts[i].y == p.y {
			return i
		}
	}
	return 0
}

//GetPoint returns Point at index.
func GetPoint(pts Points, index int) Point {
	if OutOfRange(pts, index) {
		log.Printf("Point not in dataset")
	}
	return pts[index]
}

//GetValueX returns the X coordinate of Point at index.
func GetValueX(pts Points, index int) float64 {
	if OutOfRange(pts, index) {
		log.Printf("Point not in dataset")
	}
	return pts[index].x
}

//GetValueY returns the Y coordinate of Point at index.
func GetValueY(pts Points, index int) float64 {
	if OutOfRange(pts, index) {
		log.Printf("Point not in dataset")
	}
	return pts[index].y
}

//OutOfRange returns true if index is out of range.
func OutOfRange(pts Points, index int) bool {
	if len(pts) < index+1 {
		return true
	}
	return false
}

//RemoveFromPoints removes p from Points
func RemoveFromPoints(pts Points, p Point) Points {
	index := GetIndex(pts, p)
	return append(pts[:index], pts[index+1:]...)
}

//RemovePointAt removes Point at index from Points.
func RemovePointAt(pts Points, index int) Points {
	return append(pts[:index], pts[index+1:]...)

}

//ReverseIndexes flips the array indices.
func ReverseIndexes(pts Points) Points {
	for i := range pts {
		pts[i] = pts[len(pts)-1-i]
	}
	return pts
}

//LimitTo limits Points to x and y limits
func LimitTo(pts Points, xUpper, xDown, yUpper, yDown float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.x > xUpper || p.x < xDown || p.y > yUpper || p.y < yDown {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToXUpper limits Points to an upper X limit.
func LimitToXUpper(pts Points, xUpper float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.x > xUpper {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToYUpper limits Points to an upper X limit.
func LimitToYUpper(pts Points, yUpper float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.y > yUpper {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToXDown limits Points to an upper X limit.
func LimitToXDown(pts Points, xDown float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.x < xDown {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToYDown limits Points to an upper X limit.
func LimitToYDown(pts Points, yDown float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.y < yDown {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//PointInDataset returns a bool if a Point belongs to the dataset
func PointInDataset(pts Points, p Point) bool {
	for _, pi := range pts {
		if pi.x == p.x && pi.y == p.y {
			return true
		}
	}
	return false
}

//FindMaxX returns the Point with the highest X value
func FindMaxX(pts Points) float64 {
	max := 0.5
	for _, p := range pts {
		if p.x > max {
			max = p.x
		}
	}
	return max
}

//FindMaxY returns the Point with the highest Y value
func FindMaxY(pts Points) float64 {
	max := 0.5
	for _, p := range pts {
		if p.y > max {
			max = p.y
		}
	}
	return max
}

//FindMinX returns the Point with the lowest X value
func FindMinX(pts Points) float64 {
	min := 0.5
	for _, p := range pts {
		if p.x < min {
			min = p.x
		}
	}
	return min
}

//FindMinY returns the Point with the lowest Y value
func FindMinY(pts Points) float64 {
	min := 0.5
	for _, p := range pts {
		if p.y < min {
			min = p.y
		}
	}
	return min
}

//GetEigenValues returns the eigenvalues. Inputs should be eigenvectors.
func GetEigenValues(m matrix.Matrix) ([]float64, error) {
	eigens, err := matrix.CalculateEigenvectors2x2(m)
	if err != nil {
		return nil, fmt.Errorf("eigenvectors could not be calculated :%v", err)
	}
	var values []float64
	for _, v := range eigens {
		l := v.GetLength()
		values = append(values, l)
	}
	return values, nil
}

func checkPrime(x int) bool {
	for i := 1; i < x; i++ {
		if x%i == 0 {
			return false
		}
	}
	return true
}

//Factorial returns the factorial of n
func Factorial(n int) int {
	var value int
	if n > 0 {
		value = n * Factorial(n-1)
		return value
	}
	return 1
}

//ApproximateLine approxiamates the line via gradient descent
func ApproximateLine(pts Points, learningRate float64, iterations int) (k, n float64) {
	for i := 0; i < iterations; i++ {
		cost, dm, dc := PointGradient(pts, k, n)
		k += -dm * learningRate
		n += -dc * learningRate
		if (10 * i % iterations) == 0 {
			fmt.Printf("cost(%.2f,%.2f) =%.2f\n", k, n, cost)
		}
	}
	return k, n
}

//PointGradient returns the vector gradients of Points
func PointGradient(pts Points, k, n float64) (cost, dm, dc float64) {
	ps := DiscludeNan(pts)
	for i := range ps {
		dist := ps[i].y - (ps[i].x*k + n)
		cost += dist * dist
		dm += -ps[i].x * dist
		dc += -dist
	}
	l := float64(len(ps))
	return cost / l, 2 / l * dm, 2 / n * dc
}

//ErrorBetweenPoints returns the average error between est and real.
func ErrorBetweenPoints(est, real Points) float64 {
	var absError float64
	if len(real) > len(est) {
		real = RemoveFromIndexUpwards(real, len(est)-1)
	} else if len(real) < len(est) {
		est = RemoveFromIndexUpwards(est, len(real)-1)
	}
	for i := range real {
		absError += real[i].y - est[i].y
	}
	return absError / float64(len(real))
}

//BiggestIndividualError returns the highest error of individual Points.
func BiggestIndividualError(est, real Points) float64 {
	var absError float64
	var arr []float64
	if len(real) > len(est) {
		real = RemoveFromIndexUpwards(real, len(est)-1)
	} else if len(real) < len(est) {
		est = RemoveFromIndexUpwards(est, len(real)-1)
	}
	for i := range real {
		absError = real[i].y - est[i].y
		arr = append(arr, absError)
	}
	for i := len(arr); i > 0; i-- {
		for j := 1; j < i; j++ {
			if arr[j-1] > arr[j] {
				swapFloats(arr, j)
			}
		}
	}
	for k := range arr {
		fmt.Printf("At index: %d is error:  %.2f\n", k, arr[k])
	}
	return arr[len(arr)-1]
}

//SmallestIndividualError returns the smallest error of individual Points.
func SmallestIndividualError(est, real Points) float64 {
	var absError float64
	var arr []float64
	if len(real) > len(est) {
		real = RemoveFromIndexUpwards(real, len(est)-1)
	} else if len(real) < len(est) {
		est = RemoveFromIndexUpwards(est, len(real)-1)
	}
	for i := range real {
		absError = real[i].y - est[i].y
		arr = append(arr, absError)
	}
	for i := len(arr); i > 0; i-- {
		for j := 1; j < i; j++ {
			if arr[j-1] > arr[j] {
				swapFloats(arr, j)
			}
		}
	}
	for k := range arr {
		fmt.Printf("At index: %d is error:  %.2f\n", k, arr[k])
	}
	return arr[0]

}

//SamePoints  returns true if p1 dataset is the same as the p2 dataset.
func SamePoints(p1, p2 Points) bool {
	p1, p2 = FixDifferentPointSizes(p1, p2)
	for i := range p1 {
		if p1[i].x == p2[i].x && p1[i].y == p2[i].y {
			return true
		}
	}
	return false
}

//EstimationError returns the average error between Points and the function f at same parameters.
func EstimationError(f func(x float64) float64, pts Points) float64 {
	var loss float64
	var e float64
	for _, p := range pts {
		e = p.y - f(p.x)
		loss += e
	}
	return loss / float64(len(pts))
}

//Mse returns the mean squared error between the function f and Points
func Mse(f func(x float64) float64, pts Points) float64 {
	var loss float64
	var e float64
	for _, p := range pts {
		e = math.Pow(p.y-f(p.x), 2)
		loss += e
	}
	return loss
}

//Rmse returns the root mean squared error between the function f and Points
func Rmse(f func(x float64) float64, pts Points) float64 {
	return math.Sqrt(Mse(f, pts))

}

//CrossEntropy returns the cross entropy loss
func CrossEntropy(f func(x float64) float64, pts Points) float64 {
	var loss float64
	for _, p := range pts {
		loss += p.y*math.Log(f(p.x)) + (1-p.y)*math.Log(1-p.x)
	}
	if checkIsNan(loss) {
		return 0
	}
	return loss
}

func swapFloats(floats []float64, index int) {
	val := floats[index]
	floats[index] = floats[index-1]
	floats[index-1] = val
}

//RemoveFromIndexUpwards removes all Points from index Points upwards from the array
func RemoveFromIndexUpwards(pts Points, index int) Points {
	for i := range pts {
		if index > i {
			pts = RemoveFromPoints(pts, pts[i])
		}
	}
	return pts
}

//RemoveFromIndexDownwards removes all Points from index Points downwards from the array
func RemoveFromIndexDownwards(pts Points, index int) Points {
	for i := range pts {
		if index < i {
			pts = RemoveFromPoints(pts, pts[i])
		}
	}
	return pts
}

//FixDifferentPointSizes equalizes Point sizes by shortening the longer one.
func FixDifferentPointSizes(p1, p2 Points) (Points, Points) {
	if len(p1) > len(p2) {
		p1 = RemoveFromIndexUpwards(p1, len(p2)-1)
	} else if len(p1) < len(p2) {
		p2 = RemoveFromIndexDownwards(p2, len(p1)-1)
	}
	return p1, p2
}

//ApplyProductRule returns the function via the productRule.
//Note that the input function should be of shape a*b**x
func ApplyProductRule(f func(x float64) float64, base, power float64) func(float64) float64 {
	f2 := func(x float64) float64 {
		return power * base * math.Pow(f(x), power-1)
	}
	return f2
}

//Gradient returns the gradients from startPoint to endPoint
//See GradientAt for gradient at a specific Point.
func Gradient(f func(x float64) float64, startPoint, endPoint float64) []float64 {
	var grads []float64
	step := 0.01
	for i := startPoint; i < endPoint; i += step {
		grad := (f(i+step) - f(i)) / ((i + step) - i)
		grads = append(grads, grad)
		fmt.Printf("%.2f : %.2f\n", i+1, grad)

	}
	return grads
}

//GradientAt returns the gradient of f at x
func GradientAt(f func(x float64) float64, x float64) float64 {
	step := 0.01
	grad := (f(x+step) - f(x)) / ((x + step) - x)
	return grad
}

func degreesToRadians(x float64) float64 {
	return x * math.Pi / 180
}
func radiansToDegrees(x float64) float64 {
	return x * 180 / math.Pi
}

//DoApproximation uses the ApproximateLine outputs to plot the approximation line.
func DoApproximation(pts Points, file string) error {
	x, c := ApproximateLine(pts, 0.01, 10000)
	if err := DrawApproximation(pts, x, c, file); err != nil {
		return fmt.Errorf("Drawing approximation failed. :%v", err)
	}
	return nil
}

//knn is the K-Nearest-Neighbour classification algorithm.
//You can vary the parameter k for k nearest neighbours to be selected as an estimation.
//This is binary algorithm set by points' isBlue attribute.
func Knn(k int, p Point, pts []Point) bool {
	var kValue float64
	ps := TopKEuclideans(p, pts, k)
	for _, pt := range ps {
		var v float64
		if !pt.isBlue {
			v = -1
		}
		kValue += Euclidean(p, pt) * v
	}
	if kValue > 0 {
		return true
	}
	return false
}

//Which Points have the shortest Euclidean distance to the Point p.
//It is used with KNN Algorithm.
func TopKEuclideans(p Point, pts []Point, k int) []Point {
	var eucs []float64
	var ps []Point
	for _, pt := range pts {
		euc := Euclidean(p, pt)
		eucs = append(eucs, euc)
		eucs = sort(eucs)
		eucs = eucs[:k]
	}
	for i := range eucs {
		for _, pt := range pts {
			if eucs[i] == Euclidean(p, pt) {
				ps = append(ps, pt)
			}
		}
	}
	return ps
}

func sort(fs []float64) []float64 {
	for i := len(fs); i > 0; i-- {
		for j := 1; j < i; j++ {
			if fs[j-1] > fs[j] {
				swapFs(fs, j)
			}
		}
	}
	return fs
}
func swapFs(ps []float64, index int) {
	val := ps[index]
	ps[index] = ps[index-1]
	ps[index-1] = val
}
