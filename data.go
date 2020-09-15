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
	"gonum.org/v1/plot/plotter"
)

//Point struct
//Use the boolean is and categories only if needed.
type Point struct {
	X, Y float64
	Is   bool
	Cat  Category
}

//Category used with KNN algorithm
type Category struct {
	name string
	c    color.RGBA
}

//Points is a slice of Points.
type Points []Point

//PrintPoints prints Points by line.
func PrintPoints(pts Points) {
	for i, p := range pts {
		fmt.Printf("%d : (%.2f, %.2f)\n", i+1, p.X, p.Y)
	}
}

//NewPoint returns a new Point object.
func NewPoint(x, y float64) Point {
	return Point{X: x, Y: y}
}

//PointsFromArrays returns Points from arrays of x and y coordiantes.
func PointsFromArrays(xs, ys []float64) Points {
	pts := make(Points, len(xs))
	for i := range pts {
		pts = append(pts, Point{X: xs[i], Y: ys[i]})
	}
	return pts
}

//Xs returns all x coordinates of the dataset.
func Xs(pts Points) []float64 {
	arr := make([]float64, len(pts))
	for _, p := range pts {
		arr = append(arr, p.X)
	}
	return arr
}

//Ys returns all y coordinates of the dataset.
func Ys(pts Points) []float64 {
	arr := make([]float64, len(pts))
	for _, p := range pts {
		arr = append(arr, p.Y)
	}
	return arr
}

//RandomPoint returns a random Point
func RandomPoint() Point {
	rand.Seed(time.Now().UnixNano())
	return Point{X: rand.Float64(), Y: rand.Float64()}
}

//RoundPointstoDecimals rounds all Points to decimals accuracy.
func RoundPointstoDecimals(pts Points, decimals int) Points {
	for _, p := range pts {
		p.X = roundTo(p.X, decimals)
		p.Y = roundTo(p.Y, decimals)
	}
	return pts
}

//Randomize randomizes Points to random values.
func Randomize(pts Points) Points {
	fls := randFloats(-5.0, 5.0, 2*len(pts))
	for i := range pts {
		pts[i].X = fls[i]
		pts[i].Y = fls[len(fls)-i]
	}
	return pts
}

//Standardize standardizes Points so that they are centered and the standard deviation is 1.
func Standardize(pts Points) Points {
	for i := range pts {
		pts[i].X = (pts[i].X - AverageX(pts)) / StddevX(pts)
		pts[i].Y = (pts[i].Y - AverageY(pts)) / StddevY(pts)
	}
	return pts
}

//Normalize normalizes data so that it is between the interval of 0 and 1
func Normalize(pts Points) Points {
	for i := range pts {
		pts[i].X = (pts[i].X - FindMinX(pts)) / (FindMaxX(pts) - FindMinX(pts))
		pts[i].Y = (pts[i].Y - FindMinY(pts)) / (FindMaxY(pts) - FindMinY(pts))
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
		arr = append(arr, p.X)
	}
	return arr
}

func yToArray(pts Points) []float64 {
	arr := make([]float64, len(pts))
	for _, p := range pts {
		arr = append(arr, p.Y)
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
		if p.Y == y {
			return i
		}
	}
	return 0
}
func indexOfX(pts Points, x float64) int {
	for i, p := range pts {
		if p.X == x {
			return i
		}
	}
	return 0
}

//ClosestToZero returns value closest to zero. You can use this function to  see which approximation best estimates the actual function.
func ClosestToZero(pts Points) float64 {
	zero := 10.0
	for _, f := range pts {
		abs := Absolute(f.Y)
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
		v := p.Y - value
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

//DistanceFromOrigin returns the euclidean distance from origin.
func DistanceFromOrigin(p Point) float64 {
	return Euclidean(p, Point{X: 0, Y: 0})
}

//TriangleArea returns the area of a triangle via the determinant.
func TriangleArea(p1, p2, p3 Point) float64 {
	return 1 / 2 * (p2.X*p3.Y - (p1.X*p3.Y - p3.X*p1.Y) + (p1.X*p2.Y - p2.X*p1.Y))
}

//Euclidean returns the Euclidean distance between two Points.
func Euclidean(p1, p2 Point) float64 {
	s := math.Pow((p1.X - p2.X), 2)
	k := math.Pow((p1.Y - p2.Y), 2)
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
		pts[i].X = -pts[i].X
	}
	return pts
}

//FlipOverYAxis flips Points over the Y axis.
func FlipOverYAxis(pts Points) Points {
	for i := range pts {
		pts[i].Y = -pts[i].Y
	}
	return pts
}

//Slope returns the coeffcient for linear regression
func Slope(p1, p2 Point) float64 {
	rise := p2.Y - p1.Y
	run := p2.X - p1.X
	return rise / run
}

//Intercept returns the intercept or the value of the linear function at Point 0.
func Intercept(p1, p2 Point) float64 {
	slope := Slope(p1, p1)
	return p1.Y - slope*p1.X
}

//SortPointsByX bubble sorts Points, so the Point with the highest X value is the last element of the array.
func SortPointsByX(pts Points) Points {
	for i := len(pts); i > 0; i-- {
		for j := 1; j < i; j++ {
			if pts[j-1].X > pts[j].X {
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
			if pts[j-1].Y > pts[j].Y {
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
			X: i,
			Y: f(i),
		}
		pts = append(pts, p)
	}
	return pts
}

//DefineDatasetPolynomial returns an array of Points given the inputs. The function will iterate from stPoint to endPoint with iterations.
func DefineDatasetPolynomial(f func(x float64, polynomial int) float64, stPoint, endPoint float64, iterations, polynomial int) Points {
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := Point{
			X: i,
			Y: f(i, polynomial),
		}
		pts = append(pts, p)
	}
	return pts
}

//DefineDatasetToPower  is designed to plot Points to the power datasets.
func DefineDatasetToPower(n float64, stPoint, endPoint float64, iterations int) Points {
	var pts Points
	iter := (endPoint - stPoint) / float64(iterations)
	for i := stPoint; i <= endPoint; i += iter {
		p := Point{
			X: i,
			Y: math.Pow(i, n),
		}
		pts = append(pts, p)
	}
	return pts
}

//DefineNormalDistribution defines the normal distribution via n which is the number of points, min and max, and the mean and the standard deviaton.
func DefineNormalDistribution(n int, min, max, mean, stddev float64) Points {
	var pts Points
	iter := 2 * stddev / float64(n)
	for i := (mean - stddev) - mean; i < 2*stddev+mean; i += iter {
		p := Point{
			X: i,
			Y: Gaussian(i, mean, stddev),
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
			X: min + rand.Float64()*(max-min),
			Y: min + rand.Float64()*(max-min),
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
			X: i,
			Y: f(i) * noise,
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
			X: i,
			Y: f(i, a, b, c),
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
			X: i,
			Y: Series(s, i),
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
		pts[i].X = pts[i].X + scalar
	}
	return pts
}

//ShiftDatasetOnY shifts the y coordinates by the scalar
func ShiftDatasetOnY(pts Points, scalar float64) Points {
	for i := range pts {
		pts[i].Y = pts[i].Y + scalar
	}
	return pts
}

//StretchByFactorX streches the x coordinates by the factor. Check how mean and variance change.
func StretchByFactorX(pts Points, factor float64) Points {
	for i := range pts {
		pts[i].X = pts[i].X * factor
	}
	return pts
}

//StretchByFactorY streches the x coordinates by the factor. Check how mean and variance change.
func StretchByFactorY(pts Points, factor float64) Points {
	for i := range pts {
		pts[i].Y = pts[i].Y * factor
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
			X: x,
			Y: y,
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
		vec.Slice()[0] = pts[i].X
		vec.Slice()[1] = pts[i].Y
		vectors = append(vectors, vec)
	}
	return vectors
}

//PointToXYs is a helper function
func PointToXYs(pts Points) plotter.XYs {
	xys := make(plotter.XYs, len(pts))
	for i, p := range pts {
		xys[i].X = p.X
		xys[i].Y = p.Y
	}

	return xys
}

//VectorToPoints is a helper function
func VectorToPoints(vectors []matrix.Vector) Points {
	pts := make(Points, len(vectors))
	for _, v := range vectors {
		p := Point{
			X: v.Slice()[0],
			Y: v.Slice()[1],
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
		if !checkIsNan(p.X) && !checkIsNan(p.Y) {
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
		if pts[i].X == p.X && pts[i].Y == p.Y {
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
	return pts[index].X
}

//GetValueY returns the Y coordinate of Point at index.
func GetValueY(pts Points, index int) float64 {
	if OutOfRange(pts, index) {
		log.Printf("Point not in dataset")
	}
	return pts[index].Y
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
		if p.X > xUpper || p.X < xDown || p.Y > yUpper || p.Y < yDown {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToXUpper limits Points to an upper X limit.
func LimitToXUpper(pts Points, xUpper float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.X > xUpper {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToYUpper limits Points to an upper X limit.
func LimitToYUpper(pts Points, yUpper float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.Y > yUpper {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToXDown limits Points to an upper X limit.
func LimitToXDown(pts Points, xDown float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.X < xDown {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//LimitToYDown limits Points to an upper X limit.
func LimitToYDown(pts Points, yDown float64) Points {
	ps := DiscludeNan(pts)
	for _, p := range ps {
		if p.Y < yDown {
			ps = RemoveFromPoints(ps, p)
		}
	}
	return ps
}

//PointInDataset returns a bool if a Point belongs to the dataset
func PointInDataset(pts Points, p Point) bool {
	for _, pi := range pts {
		if pi.X == p.X && pi.Y == p.Y {
			return true
		}
	}
	return false
}

//FindMaxX returns the Point with the highest X value
func FindMaxX(pts Points) float64 {
	max := 0.5
	for _, p := range pts {
		if p.X > max {
			max = p.X
		}
	}
	return max
}

//FindMaxY returns the Point with the highest Y value
func FindMaxY(pts Points) float64 {
	max := 0.5
	for _, p := range pts {
		if p.Y > max {
			max = p.Y
		}
	}
	return max
}

//FindMinX returns the Point with the lowest X value
func FindMinX(pts Points) float64 {
	min := 0.5
	for _, p := range pts {
		if p.X < min {
			min = p.X
		}
	}
	return min
}

//FindMinY returns the Point with the lowest Y value
func FindMinY(pts Points) float64 {
	min := 0.5
	for _, p := range pts {
		if p.Y < min {
			min = p.Y
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

//ErrorBetweenPoints returns the average error between est and real.
func ErrorBetweenPoints(est, real Points) float64 {
	var absError float64
	if len(real) > len(est) {
		real = RemoveFromIndexUpwards(real, len(est)-1)
	} else if len(real) < len(est) {
		est = RemoveFromIndexUpwards(est, len(real)-1)
	}
	for i := range real {
		absError += real[i].Y - est[i].Y
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
		absError = real[i].Y - est[i].Y
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
		absError = real[i].Y - est[i].Y
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
		if p1[i].X == p2[i].X && p1[i].Y == p2[i].Y {
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
		e = p.Y - f(p.X)
		loss += e
	}
	return loss / float64(len(pts))
}

//Mse returns the mean squared error between the function f and Points
func Mse(f func(x float64) float64, pts Points) float64 {
	var loss float64
	var e float64
	for _, p := range pts {
		e = math.Pow(p.Y-f(p.X), 2)
		loss += e
	}
	return loss
}

//Rmse returns the root mean squared error between the function f and Points
func Rmse(f func(x float64) float64, pts Points) float64 {
	return math.Sqrt(Mse(f, pts))

}

//RidgeRegression returns the RidgeRegression or the l2 regularization to the loss function.
func RidgeRegression(actual, pred []float64, lambda float64) float64 {
	var loss float64
	var l2 float64
	for i := range actual {

		loss += math.Pow(pred[i]-actual[i], 2)
		l2 += lambda * math.Pow(actual[i], 2)
	}
	return loss + l2
}

//LassoRegression returns the LassoRegression or the l1 regularization to the loss function.
func LassoRegression(actual, pred []float64, lambda float64) float64 {
	var loss float64
	var l1 float64
	for i := range actual {

		loss += math.Pow(pred[i]-actual[i], 2)
		l1 += lambda * math.Abs(actual[i])
	}
	return loss + l1
}

//CrossEntropy returns the cross entropy loss
func CrossEntropy(f func(x float64) float64, pts Points) float64 {
	var loss float64
	for _, p := range pts {
		loss += p.Y*math.Log(f(p.X)) + (1-p.Y)*math.Log(1-p.X)
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

//Gradients returns the gradients from startPoint to endPoint
//See GradientAt for gradient at a specific Point.
func Gradients(f func(x float64) float64, startPoint, endPoint float64) []float64 {
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

//DegreesToRadians
func DegreesToRadians(x float64) float64 {
	return x * math.Pi / 180
}

//RadiansToDegrees
func RadiansToDegrees(x float64) float64 {
	return x * 180 / math.Pi
}

//LinearRegression returns the optimal line via gradient descent and returns the coefficient and the intercept.
func LinearRegression(pts Points, lr float64, iterations int) (k, c float64) {
	for i := 0; i < iterations; i++ {
		dk, dc := PartialsForLinReg(pts, k, c)
		k += -dk * lr
		dc += -dc * lr
	}
	return k, c
}

//Cost returns the cost function.
func Cost(pts Points, k, n float64) float64 {
	var loss float64
	for _, p := range pts {
		dist := p.Y - p.X*k + n
		loss += dist * dist
	}
	return loss
}

//PartialsForLinReg returns the
func PartialsForLinReg(pts Points, k, c float64) (dk, dc float64) {
	for _, p := range pts {
		d := p.Y - (p.X*k + c)
		dk += -p.X * d
		dc += -d
	}
	n := float64(len(pts))
	return 2 / n * dk, 2 / n * dc

}

//DoApproximation uses the ApproximateLine outputs to plot the approximation line.
func DoApproximation(pts Points, learningRate float64, iterations int, file string) error {
	x, c := LinearRegression(pts, learningRate, iterations)
	if err := DrawApproximation(pts, x, c, file); err != nil {
		return fmt.Errorf("Drawing approximation failed. :%v", err)
	}
	return nil
}

//DefineLinearDataset returns points given k=slope and n=the intercept from min to max with iterations.
func DefineLinearDataset(k, n, min, max float64, iterations int) Points {
	var pts Points
	iter := (max - min) / float64(iterations)
	for i := min; i < max; i += iter {
		pts = append(pts, NewPoint(i, k*i+n))
	}
	return pts
}

//TopKEuclideans returns the closest k points to the point p.
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

func knnclosest(k int, p Point, pts []Point) string {
	return Knn(k, p, pts).name
}

//knn is the K-Nearest-Neighbour classifiCation algorithm.
//You can vary the parameter k for k nearest neighbours to be selected as an estimation.
//Knn is implemented with the help of the struct Category.
func Knn(k int, p Point, pts []Point) Category {
	var sum float64
	var fls []float64
	ps := TopKEuclideans(p, pts, k)
	for i := range ps {
		for j := range ps {
			if ps[i].Cat == ps[j].Cat {
				sum += Euclidean(ps[i], ps[j])
			}
			fls = append(fls, sum)
		}
	}
	index := maxIndex(fls)
	return distinctcategories(pts)[index]
}

func maxIndex(fls []float64) int {
	for i := range fls {
		if fls[i] == max(fls) {
			return i
		}
	}
	return 0
}
func distinctcategories(pts []Point) []Category {
	var categories []Category
	for i := range pts {
		categories = append(categories, pts[i].Cat)
	}
	return categories[:countOfcategories(pts)]
}

func countOfcategories(pts []Point) int {
	var count int
	categories := distinctcategories(pts)
	for i := range categories {
		for j := range categories {
			if categories[i].name != categories[j].name {
				count++
			}
		}
	}
	return len(categories) - count
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

//AverageX computes the average of x coordinates of the dataset
func AverageX(pts Points) float64 {
	var sum float64
	for _, p := range pts {
		sum += p.X
	}
	avg := sum / float64(len(pts))
	return avg
}

//AverageY computes the average of y coordinates of the dataset
func AverageY(pts Points) float64 {
	var sum float64
	for _, p := range pts {
		sum += p.Y
	}
	avg := sum / float64(len(pts))
	return avg
}

//VarianceX returns the variance of X coordinates
func VarianceX(pts Points) float64 {
	var sum float64
	avg := AverageX(pts)
	for _, p := range pts {
		sum += math.Pow(p.X-avg, 2)
	}
	return sum / float64(len(pts))
}

//VarianceY returns the variance of Y coordinates
func VarianceY(pts Points) float64 {
	var sum float64
	avg := AverageY(pts)
	for _, p := range pts {
		sum += math.Pow(p.Y-avg, 2)
	}
	return sum / float64(len(pts))
}

//Covariance returns the covariance of a given dataset
func Covariance(pts Points) float64 {
	var cov float64
	avgX := AverageX(pts)
	avgY := AverageY(pts)

	for _, p := range pts {
		cov += (p.X - avgX) * (p.Y - avgY)
	}
	return cov / float64(len(pts))
}

//MedianX returns the median x value of the dataset.
func MedianX(pts Points) float64 {
	pts = SortPointsByX(pts)
	if len(pts)%2 != 0 {
		return pts[len(pts)/2-1].X
	}
	return (pts[len(pts)/2-1].X + pts[len(pts)/2].X) / 2
}

//MedianY returns the median y value of the dataset.
func MedianY(pts Points) float64 {
	pts = SortPointsByY(pts)
	if len(pts)%2 != 0 {
		return pts[len(pts)/2-1].Y
	}
	return (pts[len(pts)/2-1].Y + pts[len(pts)/2].Y) / 2
}

//IndexAtMedianX returns the index of the median x value of the dataset.
func IndexAtMedianX(pts Points) int {
	for i := range pts {
		if pts[i].X == MedianX(pts) {
			return i
		}
	}
	return 0
}

//IndexAtMedianY returns the index of the median y value of the dataset.
func IndexAtMedianY(pts Points) int {
	for i := range pts {
		if pts[i].X == MedianY(pts) {
			return i
		}
	}
	return 0
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

//JaccardIndex returns the Jaccard index.
func JaccardIndex(p1, p2 Points) int {
	var sum int
	for i := range p1 {
		if (p1[i].Is && p2[i].Is) || (!p1[i].Is && !p2[i].Is) {
			sum++
		}
	}
	return sum / len(p1)
}

//Evaluation Metrics based on the bool of Points, p1 should be the predicted value and p2 the actual values.
func F1Score(p1, p2 Points) int {
	return 2 * (Precision(p1, p2) * Recall(p1, p2)) / (Precision(p1, p2) + Recall(p1, p2))
}

//Sensitivity returns the sensitivity
func Sensitivity(p1, p2 Points) int {
	tp := TruePositivies(p1, p2)
	fn := FalseNegatives(p1, p2)
	return tp / (tp + fn)
}

//Specificity returns the specificity
func Specificity(p1, p2 Points) int {
	fp := FalsePositives(p1, p2)
	tn := TrueNegatives(p1, p2)
	return fp / (fp + tn)
}

//Precision returns the precision.
func Precision(p1, p2 Points) int {
	tp := TruePositivies(p1, p2)
	fp := FalsePositives(p1, p2)
	return tp / (tp + fp)
}

//Recall returns the recall.
func Recall(p1, p2 Points) int {
	tp := TruePositivies(p1, p2)
	fn := FalseNegatives(p1, p2)
	return tp / (tp + fn)
}

//TruePositivies returns the number of true positive predicted values.
func TruePositivies(p1, p2 Points) int {
	var sum int
	for i := range p1 {
		if p1[i].Is && p2[i].Is {
			sum++
		}
	}
	return sum
}

//TrueNegatives returns the number of true negative predicted values.
func TrueNegatives(p1, p2 Points) int {
	var sum int
	for i := range p1 {
		if !p1[i].Is && !p2[i].Is {
			sum++
		}
	}
	return sum
}

//FalsePositives returns the number of false positive predicted values.
func FalsePositives(p1, p2 Points) int {
	var sum int
	for i := range p1 {
		if !p1[i].Is && p2[i].Is {
			sum++
		}
	}
	return sum
}

//FalseNegatives returns the number of false negative predicted values.
func FalseNegatives(p1, p2 Points) int {
	var sum int
	for i := range p1 {
		if p1[i].Is && !p2[i].Is {
			sum++
		}
	}
	return sum
}

//PrintMetrics prints the evaluation metrics.
func PrintMetrics(p1, p2 Points) {
	fmt.Printf("Jaccard Index: %d\n", JaccardIndex(p1, p2))
	fmt.Printf("F1 Score: %d\n", F1Score(p1, p2))
	fmt.Printf("Precision: %d\n", Precision(p1, p2))
	fmt.Printf("Recall: %d\n", Recall(p1, p2))
}
