			function testTensor()
			{

				clearOutput();

				const chessBoard = [[8, 8]];
				const twoValues = [[22,44],[66,88]];
				const splitValues = [[27,474],[987,26]];
				const shape = [2,2];
			
				const myTensor = tf.tensor(chessBoard);
				const myTensor2 = tf.tensor(twoValues);
				const myTensor3 = tf.tensor(splitValues, shape, "int32");

				const myTensor4 = tf.tensor([[47,43],[23,94],[98,31],[87,21]]);
				const myTensor5 = tf.tensor([[2,6],[7,2],[2,6],[2,7]]);

				const addTensor4and5 = myTensor4.add(myTensor5);
				const subtractTensor4and5 = myTensor4.sub(myTensor5);
				const multTensor4and5 = myTensor4.mul(myTensor5);
				const divideTensor4and5 = myTensor4.div(myTensor5);

				myTensor3.data().then(data => testTensor1(data));
				myTensor3.array().then(array => testTensor1(array[0]));
				myTensor3.array().then(array => testTensor1(array[1]));

				const myTensor6 = tf.tensor([10,20,30,40]);
				const sqTensor = myTensor6.square();
				const myTensor7 = myTensor3.reshape([4,1]);

				document.getElementById("output").innerHTML = " Tensor 1: " + myTensor;
				document.getElementById("output2").innerHTML = "Tensor 2: " + myTensor2;
				document.getElementById("output3").innerHTML = "Tensor 3: " + myTensor3;
				document.getElementById("output6").innerHTML = "Tensor 3 - Rank: " + myTensor3.rank;
				document.getElementById("output7").innerHTML = "Tensor 3 - Shape: " + myTensor3.shape;
				document.getElementById("output8").innerHTML = "Tensor 3 - Type: " + myTensor3.dtype;
				document.getElementById("output8a").innerHTML = "Tensor 4: " + myTensor4;
				document.getElementById("output8b").innerHTML = "Tensor 5: " + myTensor5;
				document.getElementById("output9").innerHTML = "Tensor 4 & 5 - ADD: " + addTensor4and5;
				document.getElementById("output10").innerHTML = "Tensor 4 & 5 - SUBTRACT: " + subtractTensor4and5;
				document.getElementById("output11").innerHTML = "Tensor 4 & 5 - MULTIPLY: " + multTensor4and5;
				document.getElementById("output12").innerHTML = "Tensor 4 & 5 - DIVIDE: " + divideTensor4and5;
				document.getElementById("output13").innerHTML = "Tensor 6 - SQUARED: " + sqTensor;
				document.getElementById("output14").innerHTML = "Tensor 7: " + myTensor7;

				function testTensor1(data)
				{
					document.getElementById("output4").innerHTML = "Tensor Test 1: " + data;
				}

			}


			function clearOutput()
			{
				document.getElementById("output").innerHTML = "";
				document.getElementById("output2").innerHTML = "";
				document.getElementById("output3").innerHTML = "";
				document.getElementById("output4").innerHTML = "";
				document.getElementById("output5").innerHTML = ""
				document.getElementById("output6").innerHTML = "";
				document.getElementById("output7").innerHTML = "";
				document.getElementById("output8").innerHTML = "";
				document.getElementById("output8a").innerHTML = "";
				document.getElementById("output8b").innerHTML = "";
				document.getElementById("output9").innerHTML = "";
				document.getElementById("output10").innerHTML = "";
				document.getElementById("output11").innerHTML = "";
				document.getElementById("output12").innerHTML = "";
				document.getElementById("output13").innerHTML = "";
				document.getElementById("output14").innerHTML = "";
				document.getElementById("output15").innerHTML = "";
				document.getElementById("output16").innerHTML = "";
				document.getElementById("output17").innerHTML = "";
				document.getElementById("output18").innerHTML = "";
			}


			function buildModel()
			{
				
				clearOutput();
				
				document.getElementById("output").innerHTML = "Area of a Circle";

				// Training data
				const range = tf.tensor([0,1,2,3,4,5,6,7,8,9,10]);
				const PI = 22/7;
				const area = range.mul(range).mul(PI);

				// Input from one layer becomes the output for the next
				const myModel = tf.sequential();

				// Create model
				myModel.add(tf.layers.dense({units:1, inputShape:[1]}));
				myModel.compile({loss:'meanSquaredError', optimizer:'sgd'});

				// Train Model
				// Every node connected to every node in preceeding layer
				myModel.fit(range, area, {epochs:500}).then(() => {myCircleFunction()});

				function myCircleFunction()
				{

					document.getElementById("output").innerHTML = "AREA OF A CIRCLE";

					const repeats = 10;
					const r = []; // Radius
					const a = []; // Area

					for (let x = 0; x <= repeats; x++)
					{
						
						let value = myModel.predict(tf.tensor([Number(x)]));
						value.data().then(y =>
						{
							r.push(x);
							a.push(Number(y));
							if (x == repeats)
							{
								displayModel(r, a);
							}
						});
						
					}

					document.getElementById("output2").innerHTML = "Radius: " + range;
					document.getElementById("output3").innerHTML = "Area: " + area;				
					
				}


				// Display model
				function displayModel(range, area)
				{

					const data = [{x:range, y:area, mode:"markers", type:"scatter"}];

					const layout =
					{
						xaxis: {range: [0,10]},
						yaxis: {range: [0,300]},
					};

					Plotly.newPlot("output18", data, layout);

				}

			}


			// Visor
			function buildVisor()
			{
				clearOutput();

				document.getElementById("output").innerHTML = "VISOR";

				const increment = 25;
				const displayArea2 = document.getElementById("output2");
				const displayArea3 = document.getElementById("output4");
				const displayArea4 = document.getElementById("output5");
				const legend = ['Base Number', 'Result'];

				const baseNumber = [];
				const result = [];

				for (let i = 0; i < 50; i++)
				{

					baseNumber[i] = {x:i, y:i};
					result[i] = 	{x:i, y:Math.random() * increment};

				}

				const resultsPage2 = {values: [baseNumber, result], legend};
				
				const resultsPage3 = [
				{index: "JAN", value: 39},{index: "FEB", value: 95}, {index: "MAR", value: 3},
				{index: "APR", value: 7}, {index: "MAY", value: 58},{index: "JUN", value: 84},
				{index: "JUL", value: 94},{index: "AUG", value: 41},{index: "SEP", value: 14},
				{index: "OCT", value: 71},{index: "NOV", value: 90},{index: "DEC", value: 79}
				];

				// You need to use the variable name values
				const values = [
				{x: 1, y: 1},
				{x: 2, y: 4},
				{x: 3, y: 9},
				{x: 4, y: 16},
				{x: 5, y: 25}
				];

				tfvis.render.scatterplot(displayArea2, resultsPage2);
				tfvis.render.scatterplot({name: "Example Scatter Graph"}, resultsPage2);
				document.getElementById("output3").innerHTML = "Sales Figures";;
				tfvis.render.barchart(displayArea3, resultsPage3);
				tfvis.render.linechart({name: "Squared Numbers"}, {values});
				
			}
	
	
			// Marathon Statistics
			function marathonStats()
			{

				clearOutput();

				document.getElementById("output").innerHTML = "Marathon Statistics";
				
				marathonData = [
					["0 - Paris",[4,47,38]],
					["1 - Dublin",[4,46,18]],
					["2 - London",[5,0,36]],
					["3 - London",[4,21,57]],
					["4 - Toronto",[4,20,41]],
					["5 - London",[3,58,30]],
					["6 - Berlin",[4,11,37]],
					["7 - Prague",[5,21,06]],
					["8 - Amsterdam",[3,49,01]],
					["9 - Sydney",[4,31,25]]
				];

				const name = "Marathons";	
				const legend = ['City', 'Time in minutes'];
				
				const cityData = [];
				const timeData = [];
				const values = [];
				
				for (let i = 0; i < marathonData.length; i++)
				{
					cityData[i] = marathonData[i][0];
					let minutes = 0;
					const marathonTime = marathonData[i][1];
					minutes = (marathonTime[0] * 60) + marathonTime[1];
					timeData[i] = minutes;
					values[i] = {x: cityData[i], y: timeData[i]};
				}
				
				document.getElementById("output2").innerHTML = "MARATHONS";
				document.getElementById("output3").innerHTML = "CITY: " + cityData;
				document.getElementById("output4").innerHTML = "MINUTES: " + timeData;
				tfvis.render.linechart({name: "Marathons"}, {values});
				
			}

			
			// Half Marathon Statistics
			function halfMarathonStats()
			{

				clearOutput();

				document.getElementById("output").innerHTML = "Half Marathon Statistics";
	
				halfMarathonData = [
					["0 - Bungay",[1,49,18]],
					["1 - Wolverhampton",[2,05,31]],
					["2 - Liverpool",[1,56,42]],
					["3 - Silverstone",[1,51,51]],
					["4 - Bungay",[1,54,24]],
					["5 - Liverpool",[1,51,40]],
					["6 - Connemara",[2,15,47]],
					["7 - Coventry",[1,46,57]],
					["8 - Liverpool",[1,44,44]],
					["9 - Newcastle",[1,45,43]],
					["10 - Coventry",[1,43,02]],
					["11 - Liverpool",[1,46,46]],
					["12 - Coventry",[1,47,07]],
					["13 - Coventry",[1,31,16]],
					["14 - Birmingham",[1,32,08]],
					["15 - Gloucester",[1,37,59]],
					["16 - Birmingham",[1,37,40]],
					["17 - Coventry",[1,36,46]],
					["18 - Coventry",[1,45,36]],
					["19 - Birmingham",[1,39,49]],
					["20 - Birmingham",[1,48,21]],
					["21 - Birmingham",[1,31,42]],
					["22 - Limerick",[1,46,54]],
					["23 - Belfast",[1,54,16]],
					["24 - Anglesey",[1,52,31]],
					["25 - Cardiff",[1,53,23]],
					["26 - Solihull",[2,10,00]],
				];

				const cityData2 = [];
				const timeData2 = [];
				const values = [];	
				
				for (let i = 0; i < halfMarathonData.length; i++)
				{
					cityData2[i] = halfMarathonData[i][0];
					let minutes = 0;
					const halfMarathonTime = halfMarathonData[i][1];
					minutes = (halfMarathonTime[0] * 60) + halfMarathonTime[1];
					timeData2[i] = minutes;
					values[i] = {x: cityData2[i], y: timeData2[i]};
				}	

				console.log(values);
				
				document.getElementById("output2").innerHTML = "HALF MARATHONS";
				document.getElementById("output3").innerHTML = "CITY: " + cityData2;
				document.getElementById("output4").innerHTML = "MINUTES: " + timeData2;	
				tfvis.render.linechart({name: "Half Marathons"}, {values});

			}

			
			// Perceptrons
			function perceptrons()
			{
				clearOutput();

				document.getElementById("output").innerHTML = "PERCEPTRONS";

				const movies = [

					/** Template
					{
						"Name": "",
						"W1": 0, // Does the film have a recognised good actor?
						"W2": 0, // Is the film less than 3 hours long?
						"W3": 0, // Is the film from a preferred genre?
						"W4": 0, // Does the film have a recognised good director?
						"W5": 0  // Is the film not from a disliked genre?
					},
					**/

					{
						"Name": "Odyssey",
						"W1": 1,
						"W2": 1,
						"W3": 0,
						"W4": 1,
						"W5": 1
					},					
					{
						"Name": "Obsession",
						"W1": 0,
						"W2": 1,
						"W3": 0,
						"W4": 0,
						"W5": 1
					},					
					{
						"Name": "I Swear",
						"W1": 0,
						"W2": 1,
						"W3": 0,
						"W4": 0,
						"W5": 1
					},					
					{
						"Name": "Dead Reckoning",
						"W1": 1,
						"W2": 1,
						"W3": 1,
						"W4": 1,
						"W5": 1
					},
					{
						"Name": "Red Eye",
						"W1": 1,
						"W2": 1,
						"W3": 1,
						"W4": 0,
						"W5": 1
					}
					
				];

				const threshold = 1.2;

				let outputText = "";

				const weights = [0, 0.8, 0.5, 0.7, 0.3, 0.4];

				for (let i = 0; i < movies.length; i++)
				{ 

					console.log(movies[i].Name, movies[i].W1, movies[i].W2, movies[i].W3, movies[i].W4, movies[i].W5 );

					const result = (
						(movies[i].W1 * weights[1]) +
						(movies[i].W2 * weights[2]) +
						(movies[i].W3 * weights[3]) +
						(movies[i].W4 * weights[4]) +
						(movies[i].W5 * weights[5])
					);
					
					outputText += "You should see " + movies[i].Name + " = " + (result > threshold) + "  /  ";
					
				}

				document.getElementById("output2").innerHTML = outputText;
			}


			// Patterns
			function patterns()
			{

				clearOutput();

				document.getElementById("output").innerHTML = "PATTERNS";
				
				const displayArea2 = document.getElementById("output5");
				const legend = ['Base Number', 'Result'];

				const baseNumber = [];
				const result = [];
				
				const randomPoints = 100;

				const aValues = [];
				const bValues = [];

				for (let i = 0; i < randomPoints; i++)
				{
	
					aValues[i] = Math.random();
					bValues[i] = Math.random();
					
					baseNumber[i] = {x:i, y:aValues[i]};
					result[i] = 	{x:i, y:bValues[i]};
					
				}
						
				document.getElementById("output2").innerHTML = "A Values: " + aValues;
				document.getElementById("output3").innerHTML = "B Values: " + bValues;
				document.getElementById("output4").innerHTML = "OUTPUT";
				const resultsPage2 = {values: [baseNumber, result], legend};
				tfvis.render.scatterplot(displayArea2, resultsPage2);
				
			}


			// Regressions
			function regressions()
			{

				clearOutput();

				document.getElementById("output").innerHTML = "REGRESSIONS";

				const heat = [10,20,30,40,50];
				const performance = [2,3,8,7,5];

				const performanceData = 
					[
						{x:heat, y:performance, mode: "markers"},
						{x:[10,50], y:[2,8], mode:"line"}
					];

				const performanceLayout =
					{
						xaxis: {range: [0,100], title: "HEAT"},
						yaxis: {range: [0,10], title: "Performance"},
						title: "Performance in different tempretures"
					};

				Plotly.newPlot("output2", performanceData, performanceLayout);

				let heatSum = 0, performanceSum = 0, heatSum2 = 0, performanceSum2 = 0;
				let count = heat.length;

				for (let i = 0, len = count; i < count; i++)
				{
					heatSum 		+= heat[i];
					performanceSum 	+= performance[i];
					heatSum2		+= heat[i] * heat[i];
					performanceSum2	+= performance[i] * performance[i];
				}

				let gradient = (count * performanceSum2 - heatSum * performanceSum) / (count * heatSum2 - heatSum * heatSum);
				let intercept = (performanceSum / count) - (gradient * heatSum) / count;

				const heatValues = [];
				const performanceValues = [];
				
				for (let x = 10; x <= 50; x += 1)
				{
					heatValues.push(x);
					performanceValues.push(x * gradient + intercept);
				}

				const performanceData2 = 
					[
						{x:heat, y:performance, mode: "markers"},
						{x:heatValues, y:performanceValues, mode:"line"}
					];

				const performanceLayout2 =
					{
						xaxis: {range: [0,100], title: "HEAT"},
						yaxis: {range: [0,10], title: "Performance"},
						title: "Performance in different tempretures - V2"
					};

				Plotly.newPlot("output3", performanceData2, performanceLayout2);
				
			}
