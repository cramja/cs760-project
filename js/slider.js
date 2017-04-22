// A wrapper around the HTML5 slider element.
// allows users to specify custom ranges.
//

function Slider(div_id, values, callback) {
    // callback should be a function which takes the new value of the slider
    // as the input
    //

    var container_div = document.getElementById(div_id);
    this.slider_id = div_id + "-" + "slider";
    this.value_id = div_id + "-" + "value";
    // TODO: there's a more idiomatic way to do this I'm sure...
    var slider_str = "<input type='range' id='" + this.slider_id
        + "' min='0' max='" + (values.length - 1) + "'/><span id='" + this.value_id + "'></span>";
    this.values = values;
    container_div.innerHTML = slider_str;

    this.slider_ele = document.getElementById(this.slider_id);
    this.value_ele = document.getElementById(this.value_id);

    this.onchange = callback;
    this.slider_ele.oninput = (function() {
        var new_value = this.values[this.slider_ele.value];
        this.value_ele.innerHTML = "" + new_value;}).bind(this);

    this.valueChanged = function() {
        // handles call back
        var new_value = this.values[this.slider_ele.value];
        this.value_ele.innerHTML = "" + new_value;
        if (this.onchange) {
            this.onchange(new_value);
        }
    };

    this.slider_ele.onchange = (function() {this.valueChanged()}).bind(this);
    this.valueChanged();
}
