package ml

import (
	"fmt"
	"math"
)

type (
	GDescent struct {
		x      *Matrix
		y      *Vector
		theta  *Vector
		alpha  float64
		lambda float64
	}
)

//create new gradient descent object
//validations are:
//1. x should be a valid matrix
//2. number of row of X should be the same as y's length
//3, number of column of X should be the same as theta's length
//4. alpha should be greater than 0 (otherwise it will learn nothing)
func NewGradientDescent(x *Matrix, y, theta *Vector, alpha float64) (*GDescent, error) {
	//1. validation
	if err := x.validate(); err != nil {
		return nil, err
	}

	//2. validation
	if x.GetRowNumber() != y.GetLength() {
		return nil, fmt.Errorf("X and Y row number are not the same")
	}

	//3. validation
	if x.GetColumnNumber()+1 != theta.GetLength() {
		return nil, fmt.Errorf("Number of X and theta features are not the same")
	}
	//add 1 input input vector
	for key, val := range x.val {
		newx := []float64{1}
		newx = append(newx, val.val...)
		x.val[key] = NewVector(newx)
	}

	//4. validation
	if alpha <= 0 {
		return nil, fmt.Errorf("Learning rate alpha should be greater than 0")
	}

	return &GDescent{
		x:     x,
		y:     y,
		theta: theta,
		alpha: alpha}, nil
}

func (gd *GDescent) AddRegularizationFactor(lambda float64) {
	gd.lambda = lambda
}

func (gd *GDescent) String() string {
	return fmt.Sprintf(`Gradient Descent Parameter
Training matrix X:
%sTraining result Y:
%s
Gradient Theta:
%s
Learning rate alpha: %.2f
Regularization factor lambda: %.2f
`, gd.x, gd.y, gd.theta, gd.alpha, gd.lambda)
}

//calculate a prediction based on theta and an input vector
//validation is only that the length of theta should be the same as input vector
//h is sigma(1..n)thetaj*xj
func (gd *GDescent) h(input *Vector) float64 {
	return gd.theta.dotProduct(input)
}

//calculate the cost func J from gradient descent struct
//formula for cost function is: 1/2m (sigma(1...m)(h(xi) - yi) ^2  + lambda. sigma(1..n)thetaj^2)
func (gd *GDescent) cost(index int) float64 {
	//calculate h(xi) - y and then use its power of 2
	res := gd.h(gd.x.getRowVector(index)) - gd.y.getSingleValue(index)
	return math.Pow(res, 2)
}

//calculate the regularization parameter
func (gd *GDescent) regParam() float64 {
	m, n := gd.y.GetLength(), gd.theta.GetLength()

	var regParam float64
	for j := 1; j <= n; j++ {
		regParam += math.Pow(gd.theta.getSingleValue(j), 2)
	}

	return regParam * gd.lambda / (2 * float64(m))
}

func (gd *GDescent) CostFunc() float64 {
	m := gd.y.GetLength()

	var result float64
	for i := 1; i <= m; i++ {
		result += gd.cost(i)
	}
	result = result / (2 * float64(m))
	regParam := gd.regParam()

	return result + regParam
}

//calculate the derivative of the cost function by gradient theta
//the formula is:
//thetaj := thetaj - alpha/m * sigma(1...m)((h(xi) - yi) * xij)
//add regularization parameter lambda/m * thetaj if j != 1
func (gd *GDescent) derivTheta(index int) float64 {
	m := gd.y.GetLength()

	var sigma float64
	for i := 1; i <= m; i++ {
		sigma += (gd.h(gd.x.getRowVector(i)) - gd.y.getSingleValue(i)) * gd.x.getRowVector(i).getSingleValue(index)
	}

	//add regularization parameter for index != 1
	if index != 1 {
		sigma += gd.lambda * gd.theta.getSingleValue(index)
	}

	return sigma / float64(m)
}

func (gd *GDescent) CalculateGrad() *Vector {
	n := gd.x.GetColumnNumber()

	var newTheta []float64
	for j := 1; j <= n; j++ {
		newTheta = append(newTheta, gd.derivTheta(j))
	}

	return NewVector(newTheta)
}

//update theta as much as itr iterrations
func (gd *GDescent) updateGrad() {
	grad := gd.CalculateGrad()
	//fmt.Println(grad)
	grad.MultiplyVariable(-1 * gd.alpha)
	gd.theta.AddVector(grad)
}

func (gd *GDescent) UpdateGrad(itr int) {
	for i := 0; i < itr; i++ {
		gd.updateGrad()
	}
}

//calculate the result as a set of y vector
func (gd *GDescent) CalculateResult(x *Matrix) (*Vector, error) {
	//first add 1's column vector to x matrix
	for key, val := range x.val {
		newx := []float64{1}
		newx = append(newx, val.val...)
		x.val[key] = NewVector(newx)
	}

	//validate both length
	if x.GetColumnNumber() != gd.theta.GetLength() {
		return nil, fmt.Errorf("Input vector dimension(%d) does not agree with theta(%d)",
			x.GetColumnNumber(), gd.theta.GetLength())
	}

	var result []float64
	for i := 1; i <= x.GetRowNumber(); i++ {
		result = append(result, gd.h(x.getRowVector(i)))
	}

	return NewVector(result), nil
}
