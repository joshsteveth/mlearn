package ml

import (
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"strings"
)

type (
	//matrix key is the column number
	//and each key's value is the value
	//but actually better use the method getVal
	Matrix struct {
		val map[int]*Vector
	}
)

///////////////////////////
////////NEW MATRIX////////
//////////////////////////

//create a new zero matrix with zeros
func NewZeroMatrix(numRow, numCol int) *Matrix {
	m := map[int]*Vector{}

	//loop from 1 ... numCol to create the key
	for i := 1; i <= numRow; i++ {
		m[i] = NewZeroVector(numCol)
	}

	return &Matrix{
		val: m,
	}
}

//create a new matrix with a constant
func NewConstantMatrix(numRow, numCol int, val float64) *Matrix {
	m := map[int]*Vector{}

	//loop from 1 ... numCol to create the key
	for i := 1; i <= numRow; i++ {
		m[i] = NewConstantVector(numCol, val)
	}

	return &Matrix{
		val: m,
	}
}

//create a new matrix with [][]int as an input
//validate the input first
func validateMatrixInput(input [][]float64) error {
	//the input can't be empty
	if len(input) == 0 {
		return fmt.Errorf("Input matrix should not be empty")
	}

	//get the first element's length
	//this should not be empty
	numCol := len(input[0])
	if numCol == 0 {
		return fmt.Errorf("Empty row vector detected")
	}

	//the other rows length should agree with this
	for i := 1; i < len(input); i++ {
		if numCol != len(input[i]) {
			return fmt.Errorf("Number of columns does not agree")
		}
	}

	return nil
}

func NewMatrix(input [][]float64) (*Matrix, error) {
	//validate the input first
	if err := validateMatrixInput(input); err != nil {
		return nil, err
	}

	var m Matrix
	m.setValue(input)
	return &m, nil
}

//Add feature to a matrix
//e.g. for number of feature = 2 it will yield:
//1, x1, x2, x1^2, x1*x2, x2^2
//number feature = 0 yield a 1 matrix
//number feature = 1 only adds a 1 column vector
//as validation, both vectors should have the same length
func NewFeatureMatrix(v1, v2 *Vector, numfeat uint) (*Matrix, error) {
	if v1.GetLength() != v2.GetLength() {
		return nil, fmt.Errorf("Length of both input vectors are not the same")
	}

	//create new constant matrix with ones
	m := NewConstantMatrix(v1.GetLength(), 1, 1)

	//now loop between 1 until numfeat
	for i := 1; i <= int(numfeat); i++ {
		//also loop from 0 to i
		for j := 0; j <= i; j++ {
			//create new variable to avoid changing value by pointer
			var v12, v22 []float64
			v12 = append(v12, v1.val...)
			v22 = append(v22, v2.val...)

			//the new column vector is multiplication from v1^(i-j) and v2(j)
			vec1, vec2 := NewVector(v12), NewVector(v22)
			vec1.PowerOf(float64(i - j))
			vec2.PowerOf(float64(j))
			vec1.MultiplyVector(vec2)

			m.addColumnVector(vec1)
		}
	}
	return m, nil
}

//load matrix from the selected path
//using encoding csv so the file should support csv formatting with comma
//all string should be parseable into float64
func LoadNewMatrix(fileName, row, col string) (*Matrix, error) {
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

	var m Matrix
	m.setValue(filtered)
	return &m, nil
}

//modify io.Reader print for matrix
func (m *Matrix) String() string {
	sprint := fmt.Sprintf("Num Column: %d\nNum Row: %d\n",
		m.GetColumnNumber(), m.GetRowNumber())

	for i := 1; i <= m.GetRowNumber(); i++ {
		sprint += fmt.Sprintf("Row %d: %s\n", i, m.val[i])
	}

	return sprint
}

//the characteristic of a matrix is that:
//for every column it should have the same row number vice versa
func (m *Matrix) validate() error {
	var numCol, numRow int

	//of course return error if the matrix has no row
	numRow = m.GetRowNumber()
	if numRow == 0 {
		return fmt.Errorf("Matrix is empty")
	}

	//also return error if the length of first value of map is 0
	//or even does not exist
	if _, ok := m.val[1]; !ok {
		return fmt.Errorf("First row element does not exist")
	} else {
		numCol = m.GetColumnNumber()
		if numCol == 0 {
			return fmt.Errorf("First row element is empty")
		}
	}

	//now loop through all the columns to make sure all of them has the same length
	for key, row := range m.val {
		if row.GetLength() != numCol {
			return fmt.Errorf("Number of element for row %d does not match", key)
		}
	}

	return nil
}

///////////////////////////
////////GET/SET DATA///////
//////////////////////////

//get number of column and row of a matrix
func (m *Matrix) GetColumnNumber() int { return m.val[1].GetLength() }
func (m *Matrix) GetRowNumber() int    { return len(m.val) }

//get a single value of a Matrix
//for example to get the 1st column and the 5th row: GetSingleValue(1,5)
func (m *Matrix) getSingleValue(row, col int) float64 {
	return m.val[row].getSingleValue(col)
}

func (m *Matrix) GetSingleValue(row, col int) (float64, error) {
	//proof the existence of the column first
	if _, ok := m.val[row]; !ok {
		return 0, fmt.Errorf("Row %d does not exist", row)
	}

	//and then validate the row slice to avoid panic
	if m.val[row].GetLength() < col {
		return 0, fmt.Errorf("Column %d does not exist", col)
	}

	//get the value from the vector
	//return error if failed

	return m.getSingleValue(row, col), nil
}

//set a single value to a Matrix
//for example to set value 3 to the 1st column and 5th row: SetSingleValue(1,5, 3)
func (m *Matrix) setSingleValue(row, col int, newVal float64) {
	m.val[row].setSingleValue(col, newVal)
}

func (m *Matrix) SetSingleValue(row, col int, newVal float64) error {
	//first get the value for the column and row
	//if error then return error
	if _, err := m.GetSingleValue(row, col); err != nil {
		return err
	}

	m.setSingleValue(row, col, newVal)

	return nil
}

//set value to a matrix
//input is a [][]int which supposed to be converted into map[int]*Vector matrix value
//all of them can't contain any empty slice
func (m *Matrix) setValue(input [][]float64) {
	res := map[int]*Vector{}
	for key, val := range input {
		/*v := NewZeroVector(len(val))
		v.SetValue(val)*/
		res[key+1] = NewVector(val)
	}

	m.val = res
}

//function to return list of row as well as column vectors
func (m *Matrix) getRowVector(row int) *Vector {
	return m.val[row]
}

func (m *Matrix) GetRowVector(row int) (*Vector, error) {
	if m.GetRowNumber() < row {
		return nil, fmt.Errorf("Index row out of range")
	}

	return m.getRowVector(row), nil
}

func (m *Matrix) GetAllRowVectors() []*Vector {
	var result []*Vector
	for key, _ := range m.val {
		result = append(result, m.getRowVector(key))
	}
	return result
}

//getting column vector is a little bit tricky
//it requires to convert the list of elements into vector
func (m *Matrix) getColumnVector(col int) *Vector {
	var result []float64
	for i := 1; i <= m.GetRowNumber(); i++ {
		result = append(result, m.getSingleValue(i, col))
	}
	return NewVector(result)
}

func (m *Matrix) GetColumnVector(col int) (*Vector, error) {
	if m.GetColumnNumber() < col {
		return nil, fmt.Errorf("Index column out of range")
	}

	return m.getColumnVector(col), nil
}

func (m *Matrix) GetAllColumnVectors() []*Vector {
	var result []*Vector

	//matrix should have at least 1 row vector to begin with
	//return empty result otherwise
	if len(m.val) == 0 {
		return result
	}

	for i := 1; i <= m.val[1].GetLength(); i++ {
		result = append(result, m.getColumnVector(i))
	}

	return result
}

///////////////////////////
////////CALCULATION///////
//////////////////////////

//add a variable into a matrix
func (m *Matrix) AddVariable(x float64) {
	for i := 1; i <= m.GetRowNumber(); i++ {
		m.val[i].AddVariable(x)
	}
}

//multiply the matrix with a variable
func (m *Matrix) MultiplyVariable(x float64) {
	for i := 1; i <= m.GetRowNumber(); i++ {
		m.val[i].MultiplyVariable(x)
	}
}

//add two Matrizes
//as validation: dimension of both should agree
//simply add the values from both with the same index
func (m *Matrix) addMatrix(m2 *Matrix) {
	for i := 1; i <= m.GetColumnNumber(); i++ {
		for j := 1; j <= m.GetRowNumber(); j++ {
			newVal := m.getSingleValue(j, i) + m2.getSingleValue(j, i)
			m.setSingleValue(j, i, newVal)
		}
	}
}

func (m *Matrix) AddMatrix(m2 *Matrix) error {
	//make sure both of the dimensions agree
	if m.GetRowNumber() != m2.GetRowNumber() {
		return fmt.Errorf("Row number does not agree")
	}
	if m.GetColumnNumber() != m2.GetColumnNumber() {
		return fmt.Errorf("Column number does not agree")
	}

	m.addMatrix(m2)
	return nil
}

//Matrix multiplication
//the order between m and m2 matters
//number of m's col and m2's row must agree
func (m *Matrix) multiply(m2 *Matrix) *Matrix {
	//create new matrix with row = m's row and column = m2's column
	res := NewZeroMatrix(m.GetRowNumber(), m2.GetColumnNumber())

	for i := 1; i <= m.GetRowNumber(); i++ {
		for j := 1; j <= m.GetColumnNumber(); j++ {
			//create dot product between row and col vectors
			rowVec := m.getRowVector(i)
			colVec := m2.getColumnVector(j)
			dp := rowVec.dotProduct(colVec)
			res.setSingleValue(i, j, dp)
		}
	}

	return res
}

func (m *Matrix) Multiply(m2 *Matrix) (*Matrix, error) {
	//validate first
	if m.GetColumnNumber() != m2.GetRowNumber() {
		return nil, fmt.Errorf("First column and second row dimenstions don't agree")
	}

	return m.multiply(m2), nil
}

//transpose a Matrix and set the result as transposed
//validate only once and then do the transpose
func (m *Matrix) transpose() *Matrix {
	//create new matrix
	//switch length of column and row
	newCol, newRow := m.GetRowNumber(), m.GetColumnNumber()
	t := NewZeroMatrix(newRow, newCol)

	for i := 1; i <= newRow; i++ {
		for j := 1; j <= newCol; j++ {
			//also switch the index from original matrix's value
			t.setSingleValue(i, j, m.getSingleValue(j, i))
		}
	}

	return t
}

func (m *Matrix) Transpose() (*Matrix, error) {
	if err := m.validate(); err != nil {
		return nil, err
	}

	return m.transpose(), nil
}

//add a vector to a matrix
//it's either row or column vector
//the dimension must agree
//- for row vector: input length must be the same as column number
//- for column vector: input length must be the same as row number
func (m *Matrix) addRowVector(v *Vector) {
	r := m.GetRowNumber()
	m.val[r+1] = v
}

func (m *Matrix) AddRowVector(v *Vector) error {
	if v.GetLength() != m.GetColumnNumber() {
		return fmt.Errorf("Vector length must be the same as number of columns")
	}

	m.addRowVector(v)
	return nil
}

func (m *Matrix) addColumnVector(v *Vector) {
	//update all vector with the index v
	for i := 1; i <= m.GetRowNumber(); i++ {
		m.getRowVector(i).AddValue(v.getSingleValue(i))
	}
}

func (m *Matrix) AddColumnVector(v *Vector) error {
	if v.GetLength() != m.GetRowNumber() {
		return fmt.Errorf("Vector length must be the same as number of rows")
	}

	m.addColumnVector(v)
	return nil
}
