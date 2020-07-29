/**
 * @param {string[]} products
 * @param {string} searchWord
 * @return {string[][]}
 */
var suggestedProducts = function(products, searchWord) {
    products.sort();
    suggestionResults = [];
    for(var i = 0; i < searchWord.length; i++){
        let searchSubstring = searchWord.slice(0,i+1);
        let suggestionSubArray = []
        for(var j = 0; j < products.length; j++){
            if(products[j].slice(0,i+1) == searchSubstring){
                suggestionSubArray.push(products[j]);
            }
            if(suggestionSubArray.length >= 3)
                {break;}
        }
        suggestionResults.push(suggestionSubArray);
    }
    return suggestionResults;
};
