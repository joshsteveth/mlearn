package ml

import (
	"fmt"
	"math"
)

type (
	LinReg struct {
		x      *Matrix
		y      *Vector
		theta  *Vector
		alpha  float64
		lambda float64
	}
)

//create new linear regression object
//validations are:
//1. x should be a valid matrix
//2. number of row of X should be the same as y's length
//3, number of column of X should be the same as theta's length
//4. alpha should be greater than 0 (otherwise it will learn nothing)
func NewLinearRegression(x *Matrix, y, theta *Vector, alpha float64) (*LinReg, error) {
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

	return &LinReg{
		x:     x,
		y:     y,
		theta: theta,
		alpha: alpha}, nil
}

func (lr *LinReg) AddRegularizationFactor(lambda float64) {
	lr.lambda = lambda
}

func (lr *LinReg) String() string {
	return fmt.Sprintf(`Linear Regression Parameter
Training matrix X:
%sTraining result Y:
%s
Gradient Theta:
%s
Learning rate alpha: %.2f
Regularization factor lambda: %.2f
`, lr.x, lr.y, lr.theta, lr.alpha, lr.lambda)
}

//calculate a prediction based on theta and an input vector
//validation is only that the length of theta should be the same as input vector
//h is sigma(1..n)thetaj*xj
func (lr *LinReg) h(input *Vector) float64 {
	return lr.theta.dotProduct(input)
}

//calculate the cost func J from gradient descent struct
//formula for cost function is: 1/2m (sigma(1...m)(h(xi) - yi) ^2  + lambda. sigma(1..n)thetaj^2)
func (lr *LinReg) cost(index int) float64 {
	//calculate h(xi) - y and then use its power of 2
	res := lr.h(lr.x.getRowVector(index)) - lr.y.getSingleValue(index)
	return math.Pow(res, 2)
}

//calculate the regularization parameter
func (lr *LinReg) regParam() float64 {
	m, n := lr.y.GetLength(), lr.theta.GetLength()

	var regParam float64
	for j := 1; j <= n; j++ {
		regParam += math.Pow(lr.theta.getSingleValue(j), 2)
	}

	return regParam * lr.lambda / (2 * float64(m))
}

func (lr *LinReg) CostFunc() float64 {
	m := lr.y.GetLength()

	var result float64
	for i := 1; i <= m; i++ {
		result += lr.cost(i)
	}
	result = result / (2 * float64(m))
	regParam := lr.regParam()

	return result + regParam
}

//calculate the derivative of the cost function by gradient theta
//the formula is:
//thetaj := thetaj - alpha/m * sigma(1...m)((h(xi) - yi) * xij)
//add regularization parameter lambda/m * thetaj if j != 1
func (lr *LinReg) derivTheta(index int) float64 {
	m := lr.y.GetLength()

	var sigma float64
	for i := 1; i <= m; i++ {
		sigma += (lr.h(lr.x.getRowVector(i)) - lr.y.getSingleValue(i)) * lr.x.getRowVector(i).getSingleValue(index)
	}

	//add regularization parameter for index != 1
	if index != 1 {
		sigma += lr.lambda * lr.theta.getSingleValue(index)
	}

	return sigma / float64(m)
}

func (lr *LinReg) CalculateGrad() *Vector {
	n := lr.x.GetColumnNumber()

	var newTheta []float64
	for j := 1; j <= n; j++ {
		newTheta = append(newTheta, lr.derivTheta(j))
	}

	return NewVector(newTheta)
}

//update theta as much as itr iterrations
func (lr *LinReg) updateGrad() {
	grad := lr.CalculateGrad()
	//fmt.Println(grad)
	grad.MultiplyVariable(-1 * lr.alpha)
	lr.theta.AddVector(grad)
}

func (lr *LinReg) UpdateGrad(itr int) {
	for i := 0; i < itr; i++ {
		lr.updateGrad()
	}
}

//calculate the result as a set of y vector
func (lr *LinReg) CalculateResult(x *Matrix) (*Vector, error) {
	//first add 1's column vector to x matrix
	for key, val := range x.val {
		newx := []float64{1}
		newx = append(newx, val.val...)
		x.val[key] = NewVector(newx)
	}

	//validate both length
	if x.GetColumnNumber() != lr.theta.GetLength() {
		return nil, fmt.Errorf("Input vector dimension(%d) does not agree with theta(%d)",
			x.GetColumnNumber(), lr.theta.GetLength())
	}

	var result []float64
	for i := 1; i <= x.GetRowNumber(); i++ {
		result = append(result, lr.h(x.getRowVector(i)))
	}

	return NewVector(result), nil
}
