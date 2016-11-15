package ml

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func newLogisticReg(alpha float64) *LReg {
	x, _ := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{2, 3, 4},
		[]float64{3, 3, 3},
	})

	y := NewVector([]float64{1, 0, 0})
	theta := NewVector([]float64{0, 0, 0})

	result, _ := NewLogisticRegression(x, y, theta, alpha)
	return result
}

func TestValidateNewLogisticRegression(t *testing.T) {
	xinput1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
	}

	x1, _ := NewMatrix(xinput1)

	xinput2 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{3, 3, 3},
	}

	x2, _ := NewMatrix(xinput2)

	y1 := NewVector([]float64{1, 2, 3})
	y2 := NewVector([]float64{1, 0, 0})

	theta1 := NewVector([]float64{1, 2, 3})
	theta2 := NewVector([]float64{1, 2, 3, 4})

	alpha1 := float64(0)
	alpha2 := float64(1)

	//1.validation
	_, err := NewLogisticRegression(&Matrix{}, y1, theta1, alpha2)
	assert.Error(t, err)

	//2.validation
	_, err = NewLogisticRegression(x1, y2, theta1, alpha2)
	assert.Error(t, err)

	//3.validation
	_, err = NewLogisticRegression(x2, y2, theta2, alpha2)
	assert.Error(t, err)

	//4.validation
	_, err = NewLogisticRegression(x2, y2, theta1, alpha1)
	assert.Error(t, err)

	//5.validation
	_, err = NewLogisticRegression(x2, y1, theta1, alpha2)
	assert.Error(t, err)

	//all correct
	_, err = NewLogisticRegression(x2, y2, theta1, alpha2)
	assert.NoError(t, err)

}

func TestLogisticRegressionCalculateResult(t *testing.T) {
	lreg := newLogisticReg(1)
	fmt.Println(lreg)

	xfalse := NewVector([]float64{1, 2})
	_, err := lreg.CalculateResult(xfalse)
	assert.Error(t, err)

	x := NewVector([]float64{1, 2, 3})
	res, err := lreg.CalculateResult(x)
	assert.NoError(t, err)
	fmt.Printf("Logistic regression result: %.5f\n", res)

	//calculate the cost function
	cost := lreg.CostFunc()
	fmt.Printf("Logistic regression cost func: %.5f\n", cost)

	//test update the gradient value
	lreg.UpdateGradient()
	fmt.Printf("New theta: \n%s\n", lreg.theta)

	//calculation of 3,3,3 now should be very close to 0
	res, _ = lreg.CalculateResult(x)
	fmt.Printf("Updated Logistic regression result: %.5f\n", res)

	//calculate the updated cost function
	cost = lreg.CostFunc()
	fmt.Printf("Updated Logistic regression cost func: %.5f\n", cost)

	fmt.Println("")
}
