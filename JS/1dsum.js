/**
 * @param {number[]} nums
 * @return {number[]}
 */
var runningSum = function(nums) {
    solutionArray = [];
    var solutionArrayElement;
    for(var i = 0; i < nums.length; i++){
        solutionArrayElement = 0;
        for(var j = 0; j <= i; j++){
            solutionArrayElement += nums[j];
        }
        solutionArray.push(solutionArrayElement);
    }
    return solutionArray;
};
