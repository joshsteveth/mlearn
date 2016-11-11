package ml

import (
	"fmt"
	"strings"
)

type (
	Vector struct {
		val []float64
	}
)

//create a new vector matrix with zeros
func NewZeroVector(numElem int) *Vector {
	v := []float64{}

	//loop from 1...numElem to create the key
	for i := 0; i < numElem; i++ {
		v = append(v, float64(0))
	}

	return &Vector{val: v}
}

//crate a new vector matrix with an array of int as input
func NewVector(input []float64) *Vector {
	return &Vector{val: input}
}

//get number of element of a vector
func (v *Vector) GetLength() int { return len(v.val) }

func (v *Vector) String() string {
	var numList []string
	for _, val := range v.val {
		numList = append(numList, fmt.Sprintf("%.2f", val))
	}

	return fmt.Sprintf("[%s]", strings.Join(numList, " , "))
}

//get a single element of the Vector
//range is from 1 ... v.GetLength()
//return error if index is out of range
func (v *Vector) getSingleValue(index int) float64 {
	return v.val[index-1]
}

func (v *Vector) GetSingleValue(index int) (float64, error) {
	if v.GetLength() < index {
		return 0, fmt.Errorf("Index is out of range")
	}

	return v.getSingleValue(index), nil
}

//set a single element of the Vector
//range is from 1...v.GetLength()
func (v *Vector) setSingleValue(index int, newVal float64) {
	v.val[index-1] = newVal
}

func (v *Vector) SetSingleValue(index int, newVal float64) error {
	//first get the old value as validation
	if _, err := v.GetSingleValue(index); err != nil {
		return err
	}

	v.setSingleValue(index, newVal)
	return nil
}

//set value from a slice of integer
//validate the length and return error if they both don't match
func (v *Vector) SetValue(input []float64) error {
	if len(input) != v.GetLength() {
		return fmt.Errorf("Length of input does not match with vector length")
	}

	v.val = input
	return nil
}

//add a variable into the vector
func (v *Vector) AddVariable(x float64) {
	for i := 1; i <= v.GetLength(); i++ {
		v.setSingleValue(i, v.getSingleValue(i)+x)
	}
}

//multiply vector with a variable
func (v *Vector) MultiplyVariable(x float64) {
	for i := 1; i <= v.GetLength(); i++ {
		v.setSingleValue(i, v.getSingleValue(i)*x)
	}
}

//add a vector to a vector
//as validation both dimensions must agree
func (v *Vector) addVector(v2 *Vector) {
	for i := 1; i <= v.GetLength(); i++ {
		v.setSingleValue(i, v.getSingleValue(i)+v2.getSingleValue(i))
	}
}

func (v *Vector) AddVector(v2 *Vector) error {
	if v.GetLength() != v2.GetLength() {
		return fmt.Errorf("Dimensions of both vectors don't agree")
	}

	v.addVector(v2)
	return nil
}
