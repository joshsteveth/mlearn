package ml

import (
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"math"
	"strings"
)

type (
	Vector struct {
		val []float64
	}
)

////////////////////////////
////////NEW VECTOR/////////
//////////////////////////

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

//load vector from a selected path
//using encoding csv so the file should support csv formatting with comma
//all string should be parseable into float64
//column should be a single value (can only use ":" if there's only single column)
func LoadNewVector(fileName, row, col string) (*Vector, error) {
	//load the filename and convert it into [][]string first
	file, err := ioutil.ReadFile(fileName)
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(strings.NewReader(string(file)))
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	//convert the [][]string into [][]float first
	floats, err := convertCSVToFloat64(records)
	if err != nil {
		return nil, err
	}

	//filter the result (and also validate it)
	filtered, err := filterInputByCat(floats, row, col)
	if err != nil {
		return nil, err
	}

	if len(filtered) == 0 || len(filtered[0]) != 1 {
		return nil, fmt.Errorf("Vector should have exactly 1 column")
	}

	var result []float64
	//select all col into new []float64
	for _, f := range filtered {
		result = append(result, f[0])
	}

	return &Vector{val: result}, nil
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

//add a value to the end of a vector
func (v *Vector) AddValue(input float64) {
	v.val = append(v.val, input)
}

///////////////////////////
////////CALCULATOR////////
//////////////////////////

//make vector support a simple basic general calculation which modify the value
func (v *Vector) Calculate(op func(float64) float64) {
	for i := 1; i <= v.GetLength(); i++ {
		v.setSingleValue(i, op(v.getSingleValue(i)))
	}
}

//collection of functions for calculate

//add a single variable
func (v *Vector) addVar(n float64) func(float64) float64 {
	return func(x float64) float64 {
		return x + n
	}
}

func (v *Vector) AddVariable(n float64) {
	v.Calculate(v.addVar(n))
}

//multiply with a single variable
func (v *Vector) multiplyVar(n float64) func(float64) float64 {
	return func(x float64) float64 {
		return x * n
	}
}

func (v *Vector) MultiplyVariable(n float64) {
	v.Calculate(v.multiplyVar(n))
}

//get a power of n
//for example pow(2) from 3 is 9
func (v *Vector) powerOf(n float64) func(float64) float64 {
	return func(x float64) float64 {
		return math.Pow(x, n)
	}
}

func (v *Vector) PowerOf(n float64) {
	v.Calculate(v.powerOf(n))
}

//get the natural logarith
func (v *Vector) log() func(float64) float64 {
	return func(x float64) float64 {
		return math.Log(x)
	}
}

func (v *Vector) Log() {
	v.Calculate(v.log())
}

//get the decadic logarithm
func (v *Vector) log10() func(float64) float64 {
	return func(x float64) float64 {
		return math.Log10(x)
	}
}

func (v *Vector) Log10() {
	v.Calculate(v.log10())
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

//multiplication between vectors v and v2
//e.g. res := v[1]*v2[1] + v[2]*v2[2] + ... + v[n]*v2[n]
func (v *Vector) dotProduct(v2 *Vector) float64 {
	var result float64
	for i := 1; i <= v.GetLength(); i++ {
		result += v.getSingleValue(i) * v2.getSingleValue(i)
	}
	return result
}

func (v *Vector) DotProduct(v2 *Vector) (float64, error) {
	//return if length of both vector does not agree
	if v.GetLength() != v2.GetLength() {
		return 0, fmt.Errorf("Dimensions of both vectors don't agree")
	}

	return v.dotProduct(v2), nil
}
