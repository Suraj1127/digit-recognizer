var $canvas = $("canvas");

var context = $canvas[0].getContext("2d");
var lastEvent;
var mouseDown = false;

context.fillStyle = "black";
context.fillRect(0, 0, $canvas.get(0).width, $canvas.get(0).height);

$canvas.mousedown(function(e) {
    lastEvent = e;
    mouseDown = true;

}).mousemove(function(e) {
    if (mouseDown) {
        context.beginPath();
        context.moveTo(lastEvent.offsetX, lastEvent.offsetY);
        context.lineTo(e.offsetX, e.offsetY);
        context.strokeStyle = 'white';
        context.lineWidth = 10;
        context.lineCap = 'round';
        context.stroke();
        lastEvent = e;
    }
}).mouseup(function() {
    mouseDown = false;
}).mouseleave(function() {
    $canvas.mouseup();
});
$('#predict').click(function(){
    $('#data_url').val($canvas.get(0).toDataURL());
    $('#data-url').submit()
});

$(function() {


    // This function gets cookie with a given name
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');


    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    function sameOrigin(url) {
        // test that a given url is a same-origin URL
        // url could be relative or scheme relative or absolute
        var host = document.location.host; // host + port
        var protocol = document.location.protocol;
        var sr_origin = '//' + host;
        var origin = protocol + sr_origin;
        // Allow absolute or scheme relative URLs to same origin
        return (url == origin || url.slice(0, origin.length + 1) == origin + '/') ||
            (url == sr_origin || url.slice(0, sr_origin.length + 1) == sr_origin + '/') ||
            // or any other URL that isn't scheme relative or absolute i.e relative.
            !(/^(\/\/|http:|https:).*/.test(url));
    }

    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && sameOrigin(settings.url)) {
                // Send the token to same-origin, relative URLs only.
                // Send the token only if the method warrants CSRF protection
                // Using the CSRFToken value acquired earlier
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

});

// predict on predict button click
$('#data-url').on('submit', function(event){
    event.preventDefault();
    predict();
});

// new canvas on new button click
$('#newPrediction').on('click', function(event){
    $('#prediction').html('');
    context.fillStyle = "black";
    context.fillRect(0, 0, $canvas.get(0).width, $canvas.get(0).height);
    $('#prediction').css({'padding': '0px', 'border': '0px'});
});

// AJAX for predicting
function predict() {
    $.ajax({
        url : "/canvas/", // the endpoint
        type : "POST", // http method
        data : { data_url : $('#data_url').val() }, // data sent with the post request

        // handle a successful response
        success : function(response) {
            $('#data_url').val(''); // remove the value from the input
            $('#prediction').css({'border': '1px solid black', 'padding': '8px'});
            $('#prediction').text('Prediction: ' + response);
        },

        // handle a non-successful response
        error : function(xhr,errmsg,err) {
            $('#results').html("<div class='alert-box alert radius' data-alert>Oops! We have encountered an error: "+errmsg+
                " <a href='#' class='close'>&times;</a></div>"); // add the error to the dom
            console.log(xhr.status + ": " + xhr.responseText); // provide a bit more info about the error to the console
        }
    });
};