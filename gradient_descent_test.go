package ml

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGradientDescentCostFunc(t *testing.T) {
	//t.Skip()
	file := "data3.csv"

	x, _ := LoadNewMatrix(file, "1:80", "1")
	y, _ := LoadNewVector(file, "1:80", "2")
	theta := NewZeroVector(x.GetColumnNumber() + 1)
	alpha := float64(0.02)

	gd, err := NewGradientDescent(x, y, theta, alpha)
	assert.NoError(t, err)

	cost := gd.CostFunc()
	fmt.Printf("Gradient descent without regularization cost func: %.5f\n", cost)

	grad := gd.CalculateGrad()
	fmt.Printf("Gradient descent without regularization grad: %s\n", grad)

	numIt := 250
	timeNow := time.Now()
	gd.UpdateGrad(numIt)
	fmt.Printf("Time needed for %d iterations: %.2fs\n", numIt, time.Since(timeNow).Seconds())
	fmt.Printf("New gradient descent without regularization theta: %s\n", gd.theta)
	fmt.Printf("New gradient descent without regularization cost func: %.5f\n", gd.CostFunc())

	//use the remaining 20 for verification
	xverif, _ := LoadNewMatrix(file, "81:100", "1")
	yverif, _ := LoadNewVector(file, "81:100", "2")
	thetaverif := gd.theta
	gdverif, _ := NewGradientDescent(xverif, yverif, thetaverif, 1)

	//var predictTrue, predictFalse int
	var totalErr float64
	for i := 1; i <= xverif.GetRowNumber(); i++ {
		xvec, _ := xverif.GetRowVector(i)
		res := gdverif.h(xvec)
		realVal := yverif.getSingleValue(i)
		err := math.Abs(res - realVal)
		totalErr += err

		fmt.Printf("Result[%.2f] : %.2f | y = %.2f | error: %.2f\n",
			xvec.getSingleValue(2), res, realVal, err)
	}
	fmt.Printf("Average error: %.2f\n", totalErr/float64(xverif.GetRowNumber()))
	fmt.Println("")
}
