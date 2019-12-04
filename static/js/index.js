$(document).ready(()=>{
    $('#answer').hide();

    $('#get_ans').click(()=>{
        let context=$('#context').val();
        let question=$('#question').val();

        $.ajax({
            url: '/',
            type: 'post',
            data: {'context': context, 'question': question}
        }).done((res)=>{
            console.log(res);
            if(res.ans!="")
                $('#answer').html(`<b>Answer:</b> ${res.ans}`);
            else
                $('#answer').html(`<b>Answer:</b> (Not possible to answer)`);
            $('#answer').show();

        });



    });


})