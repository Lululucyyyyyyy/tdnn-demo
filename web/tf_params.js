var mean = tf.zeros([16, 15]);
var std = tf.ones([16, 15])

console.log('loaded mean and std');
tf.print(mean)
tf.print(std)

var model = await tf.loadLayersModel('tfjs_model/model.json');

console.log('loaded model')
console.log(model);