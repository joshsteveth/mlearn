//v1: supports only single class classification without regularization
package ml

import (
	"fmt"
	"math"
)

type (
	LReg struct {
		x     *Matrix
		y     *Vector
		theta *Vector
		alpha float64
	}
)

//create new logistic regression object
//validations are:
//1. x should be a valid matrix
//2. number of row of X should be the same as y's length
//3, number of column of X should be the same as theta's length
//4. alpha should be greater than 0 (otherwise it will learn nothing)
//5. all member of y should be either 1 or 0
func NewLogisticRegression(x *Matrix, y, theta *Vector, alpha float64) (*LReg, error) {
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

	//5. validation
	for i := 1; i <= y.GetLength(); i++ {
		val := y.getSingleValue(i)
		if val != float64(0) && val != float64(1) {
			return nil, fmt.Errorf("Value of y should be either 0 or 1")
		}
	}

	return &LReg{x, y, theta, alpha}, nil
}

func (lr *LReg) String() string {
	return fmt.Sprintf(`Logistic Regression Parameter
Training matrix X: 
%sTraining result Y: 
%s
Gradient Theta: 
%s
Learning rate alpha: %.2f
`, lr.x, lr.y, lr.theta, lr.alpha)
}

//calculate a prediction based on theta and an input vector
//validation is only that the length of theta should be same as input vector
//h = 1 / (1 + e^(-theta * x))
func (lr *LReg) h(input *Vector) float64 {
	return sigm(lr.theta.dotProduct(input))
}

func (lr *LReg) CalculateResult(input *Vector) (float64, error) {
	//validate both length
	if input.GetLength() != lr.theta.GetLength() {
		return 0, fmt.Errorf("Input vector dimension(%d) does not agree with theta(%d)",
			input.GetLength(), lr.theta.GetLength())
	}

	return lr.h(input), nil
}

//calculate the Cost Func J from logistic regression struct
//validation should already be done when initiated
func (lr *LReg) cost(index int) float64 {
	y := lr.y.getSingleValue(index)
	x := lr.x.getRowVector(index)
	switch y {
	case 0:
		return math.Log(1-lr.h(x)) * -1
	case 1:
		return math.Log(lr.h(x)) * -1
	}

	return 0
}

func (lr *LReg) CostFunc() float64 {
	m := lr.y.GetLength()

	var result float64
	for i := 1; i <= m; i++ {
		result += lr.cost(i)
	}

	return result / float64(m)
}

//calculate the derivative of the cost function by gradient theta
//immediately update the value with the new calculated gradient
//the formula is:
//thetaj := thetaj - alpha/m  * sigma(i..m)((h(xi) - yi)) * xij)
func (lr *LReg) derivTheta(index int) float64 {
	//calculate deriv from thetaj (j = index)
	m := lr.y.GetLength()

	var sigma float64
	for i := 1; i <= m; i++ {
		sigma += (lr.h(lr.x.getRowVector(i)) - lr.y.getSingleValue(i)) * lr.x.getRowVector(i).getSingleValue(index)
	}

	return sigma / float64(m)
}

func (lr *LReg) CalculateGrad() *Vector {
	n := lr.x.GetColumnNumber()

	var newTheta []float64
	for j := 1; j <= n; j++ {
		//calculate the derivative from index j
		newTheta = append(newTheta, lr.derivTheta(j))
	}

	return NewVector(newTheta)
}

//update the theta as much as itr iterrations
func (lr *LReg) updateGrad() {
	grad := lr.CalculateGrad()
	grad.MultiplyVariable(-1 * lr.alpha)
	lr.theta.AddVector(grad)

	fmt.Println("cost func: ", lr.CostFunc())
}

func (lr *LReg) UpdateGrad(itr int) {
	for i := 0; i < itr; i++ {
		lr.updateGrad()
	}
}
